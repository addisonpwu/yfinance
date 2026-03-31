"""
Base Repository Class for Database Operations
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Type, Optional, List
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.database import Base
from src.utils.exceptions import RecordNotFoundException, DatabaseOperationException
from src.utils.logger import LoggerManager


ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(ABC, Generic[ModelType]):
    """
    Base repository class providing common CRUD operations

    All specific repositories should inherit from this class
    and implement abstract methods as needed
    """

    def __init__(self, session: AsyncSession, model: Type[ModelType]):
        """
        Initialize repository with session and model class

        Args:
            session: AsyncSession from dependency injection
            model: ORM model class
        """
        self.session = session
        self.model = model
        self.logger = LoggerManager.get_logger(f"repository.{model.__name__.lower()}")

    async def get_by_id(self, id: int) -> Optional[ModelType]:
        """
        Get model instance by ID

        Args:
            id: Primary key value

        Returns:
            Model instance or None if not found
        """
        try:
            result = await self.session.execute(
                select(self.model).where(self.model.id == id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting {self.model.__name__} by id {id}: {e}")
            raise DatabaseOperationException(
                f"Failed to get {self.model.__name__} by id: {e}"
            )

    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[str] = None,
        descending: bool = False,
    ) -> List[ModelType]:
        """
        List model instances with pagination

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Column name to order by
            descending: Sort in descending order if True

        Returns:
            List of model instances
        """
        try:
            query = select(self.model).offset(skip).limit(limit)

            if order_by:
                order_column = getattr(self.model, order_by, None)
                if order_column is not None:
                    query = query.order_by(
                        order_column.desc() if descending else order_column.asc()
                    )

            result = await self.session.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            self.logger.error(f"Error listing {self.model.__name__}: {e}")
            raise DatabaseOperationException(
                f"Failed to list {self.model.__name__}: {e}"
            )

    async def count(self) -> int:
        """
        Count total number of records

        Returns:
            Total count
        """
        try:
            result = await self.session.execute(
                select(func.count()).select_from(self.model)
            )
            return result.scalar_one()
        except Exception as e:
            self.logger.error(f"Error counting {self.model.__name__}: {e}")
            raise DatabaseOperationException(
                f"Failed to count {self.model.__name__}: {e}"
            )

    async def create(self, model: ModelType) -> ModelType:
        """
        Create a new model instance

        Args:
            model: Model instance to create

        Returns:
            Created model instance with ID populated
        """
        try:
            self.session.add(model)
            await self.session.flush()
            await self.session.refresh(model)
            self.logger.info(f"Created {self.model.__name__} with id {model.id}")
            return model
        except Exception as e:
            self.logger.error(f"Error creating {self.model.__name__}: {e}")
            raise DatabaseOperationException(
                f"Failed to create {self.model.__name__}: {e}"
            )

    async def delete(self, id: int) -> bool:
        """
        Delete a model instance by ID

        Args:
            id: Primary key value

        Returns:
            True if deleted, raises exception if not found
        """
        try:
            instance = await self.get_by_id(id)
            if instance is None:
                raise RecordNotFoundException(
                    f"{self.model.__name__} with id {id} not found"
                )

            await self.session.delete(instance)
            await self.session.flush()
            self.logger.info(f"Deleted {self.model.__name__} with id {id}")
            return True
        except RecordNotFoundException:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting {self.model.__name__} with id {id}: {e}")
            raise DatabaseOperationException(
                f"Failed to delete {self.model.__name__}: {e}"
            )

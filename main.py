
import argparse
import os
from datetime import datetime
from src.core.services.analysis_service import run_analysis
from src.config.settings import config_manager, SPEED_MODE_PRESETS

# 尝试导入配置验证模块
try:
    from src.config.config_validator import validate_startup, get_secrets_manager, HAS_PYDANTIC
    HAS_VALIDATOR = True
except ImportError as e:
    HAS_VALIDATOR = False
    HAS_PYDANTIC = False
    print(f"⚠️  配置验证模块导入失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="靈活的股票篩選器，支援多種策略")
    parser.add_argument('--market', type=str, required=True, choices=['US', 'HK'], help="要分析的市場 (US 或 HK)")
    parser.add_argument('--no-cache-update', action='store_true', help="跳過緩存更新，直接使用現有緩存數據")
    parser.add_argument('--skip-strategies', action='store_true', help="跳過策略篩選，所有股票都進行AI分析")
    parser.add_argument('--symbol', type=str, help="指定分析單一股票代碼（例如：0017.HK）")
    parser.add_argument('--interval', type=str, default='1d', choices=['1d', '1h', '1m'], help="數據時段類型：1d（日線，默認）、1h（小時線）、1m（分鐘線）")
    parser.add_argument('--model', type=str, default=None, help="AI分析模型 (iFlow: deepseek-v3.2; NVIDIA: z-ai/glm5; Gemini: gemini-2.5-flash; 或 'all')")
    parser.add_argument('--provider', type=str, default='iflow', 
                        help="AI 提供商 (iflow, nvidia, gemini，可用逗号分隔多选，如 'iflow,nvidia,gemini')")
    parser.add_argument('--speed', type=str, default=None, choices=['fast', 'balanced', 'safe'], 
                        help=f"速度模式: fast(快速), balanced(平衡,默認), safe(安全)")
    parser.add_argument('--skip-validation', action='store_true', help="跳過啟動時的配置驗證")
    args = parser.parse_args()
    
    # 解析提供商列表（支持逗号分隔）
    providers = [p.strip().lower() for p in args.provider.split(',')]
    valid_providers = ['iflow', 'nvidia', 'gemini']
    invalid_providers = [p for p in providers if p not in valid_providers]
    if invalid_providers:
        print(f"❌ 无效的 AI 提供商: {invalid_providers}，有效选项: {valid_providers}")
        return
    providers = providers if providers else ['iflow']
    
    # 根据首个提供商自动选择默认模型
    if args.model is None:
        from src.config.constants import DEFAULT_AI_PROVIDERS
        first_provider = providers[0]
        args.model = DEFAULT_AI_PROVIDERS.get(first_provider, {}).get('default_model', 'deepseek-v3.2')
    
    # 启动验证（检查配置和敏感信息）
    if HAS_VALIDATOR and not args.skip_validation:
        try:
            # 只验证配置文件，敏感信息（如 API Key）在需要时检查
            from src.config.config_validator import get_config_validator
            validator = get_config_validator()
            validator.load_and_validate()
            
            # 检查敏感信息配置状态
            secrets = get_secrets_manager()
            
            # 检查每个提供商的 API Key
            for provider in providers:
                if provider == 'nvidia':
                    if not secrets.is_configured('NVIDIA_API_KEY'):
                        print("⚠️  NVIDIA_API_KEY 未配置，NVIDIA AI 分析功能将不可用")
                        print("   请创建 .env 文件并设置 NVIDIA_API_KEY")
                elif provider == 'gemini':
                    if not secrets.is_configured('GEMINI_API_KEY'):
                        print("⚠️  GEMINI_API_KEY 未配置，Gemini AI 分析功能将不可用")
                        print("   请创建 .env 文件并设置 GEMINI_API_KEY")
                        print("   获取密钥: https://ai.google.dev/")
                else:
                    if not secrets.is_configured('IFLOW_API_KEY'):
                        print("⚠️  IFLOW_API_KEY 未配置，iFlow AI 分析功能将不可用")
                        print("   请创建 .env 文件并设置 IFLOW_API_KEY")
        except ValueError as e:
            print(f"❌ 配置验证失败: {e}")
            print("   请检查 config.json 文件格式")
            return
    
    # 应用速度模式
    if args.speed:
        config_manager.apply_speed_mode(args.speed)
    
    print(f"--- 開始對 {args.market.upper()} 市場進行分析 ---")
    if args.no_cache_update:
        print(f"--- 已啟用快速模式：跳過緩存更新 ---")
    if args.skip_strategies:
        print(f"--- 已啟用跳過策略模式：所有股票都進行AI分析 ---")
    if args.symbol:
        print(f"--- 分析指定股票: {args.symbol} ---")
    print(f"--- 數據時段類型: {args.interval} ---")
    print(f"--- AI 提供商: {providers} ---")
    if len(providers) > 1:
        print(f"--- AI分析模型: {args.model} (将使用首个模型) ---")
    else:
        print(f"--- AI分析模型: {args.model} ---")
    
    # 显示当前速度配置
    config = config_manager.get_config()
    print(f"--- 速度配置: 並行={config.api.max_workers}, 延遲={config.api.base_delay}s ---")

    # 生成报告文件名（不含扩展名，由 ReportWriter 处理）
    today_str = datetime.now().strftime('%Y-%m-%d')
    output_filename = f"{args.market.lower()}_stocks_{today_str}"

    final_list = run_analysis(
        args.market,
        force_fast_mode=args.no_cache_update,
        skip_strategies=args.skip_strategies,
        symbol_filter=args.symbol,
        interval=args.interval,
        model=args.model,
        providers=providers,
        output_filename=output_filename
    )

    print("\n--- 最終篩選結果 ---")
    if final_list:
        # 打印股票代码列表
        print(f"共筛选出 {len(final_list)} 只股票:")
        for stock in final_list:
            print(f"  ✅ {stock['symbol']}")
        print(f"\n--- 完整報告已儲存至 reports/ 目錄 ---")
    else:
        print("在指定的市場中，沒有找到符合任何策略的股票。")

if __name__ == '__main__':
    main()

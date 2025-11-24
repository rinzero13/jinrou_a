# src/modules/policy_generator.py (修正案)

from aiwolf_nlp_common.packet import Role
from utils.agent_logger import AgentLogger

class UtterancePolicyGenerator:
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    # メソッド名を変更: 最終発言ではなく、計画のための指示を生成する
    def get_planning_prompt_instructions(self, role: Role) -> str: 
        """M3: 主張の核の決定と応答方針をJSONで決定させるための指示を生成する。"""
        instructions = ""
        instructions += (
            "【M3: 初期計画と議論の焦点決定】\n"
            "あなたは、このターンにおける**議論の最優先目標（主張の核）**と、**直前の発話に対する応答の要否**を決定する戦略AIです。\n"
            
            "### 1. 主張の核（コア目標）の定義\n"
            "あなたの役職の勝利目標とゲーム状況に基づき、このターンで達成すべき**最優先の戦略的目標**を簡潔に定義してください。\n"
            
            "### 2. 他プレイヤーへの応答の要否と方針の決定\n"
            "直前の他プレイヤーの発話があなたへの直接的な質問や告発、またはあなたの主張の核に大きく影響するかを判断し、応答が必要な場合は 'RESPOND_CRITICALLY' を、コア目標を優先する場合は 'PRIORITIZE_CORE' を選択してください。\n"
            
            "\n【出力形式】:\n"
            "思考プロセスは不要です。必ず以下の**JSON形式**で、**唯一のオブジェクト**を出力してください。\n"
            "```json\n"
            "{\n"
            "  \"core_goal\": \"[最優先で達成すべき戦略的目標を簡潔に記述]\",\n"
            "  \"response_policy\": \"RESPOND_CRITICALLY\" | \"PRIORITIZE_CORE\",\n"
            "  \"response_target_id\": \"[応答対象のプレイヤーID。不要な場合は 'NONE']\"\n"
            "}\n"
            "```\n"
        )
        self.logger.debug("M3 Planning instructions generated for external call.")
        return instructions
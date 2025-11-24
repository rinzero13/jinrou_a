# src/modules/policy_generator.py

from aiwolf_nlp_common.packet import Role
from utils.agent_logger import AgentLogger

class UtterancePolicyGenerator:
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def get_policy_instructions(self, role: Role) -> str:
        """M3の指示をLLMプロンプトに追加する。"""
        instructions = ""
        # M3: 発話方針決定・応答生成の指示
        instructions += (
            "あなたの発言は、単なる意見表明ではなく、「主張の核」と「他プレイヤーへの応答」を両立した議論として機能しなければなりません。\n"
            "【議論の最優先目標】: 他者の発言を無視せず、必ず直近の論点に言及しつつ、**自分の最優先の主張の核**を論理的に展開してください。\n"
            "【発言の品質】: 一方的な発言は避け、具体的なプレイヤーの言動を根拠として挙げてください。\n"
        )
        self.logger.debug("M3 Policy instructions generated.")
        return instructions
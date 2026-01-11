from typing import List, Dict
from aiwolf_nlp_common.packet import Role

class ActionModule:
    """
    信念モデルの役職確率と対話要約を用いて、具体的な行動（投票・占い・襲撃）を決定する。
    """
    def __init__(self, logger):
        self.logger = logger

    def decide_vote_target(self, alive_agents: List[str], role_probs: Dict[str, Dict[str, float]], my_role: Role, my_id: str) -> str:
        """投票先を決定：村人陣営なら人狼確率が高い人、人狼陣営なら役職者らしい人を優先"""
        target = None
        max_score = -1.0

        for agent_id in alive_agents:
            if agent_id == my_id: continue
            
            p = role_probs.get(agent_id, {})
            # 村人陣営視点：人狼（WEREWOLF）または狂人（POSSESSED）の合算確率が高い順
            if my_role not in [Role.WEREWOLF, Role.POSSESSED]:
                score = p.get("WEREWOLF", 0.0) + p.get("POSSESSED", 0.0)
            # 人狼陣営視点：村人陣営の役職者（SEER, BODYGUARD）らしい順
            else:
                score = p.get("SEER", 0.0) + p.get("BODYGUARD", 0.0) + p.get("VILLAGER", 0.0)

            if score > max_score:
                max_score = score
                target = agent_id
        
        return target or alive_agents[0]

    def decide_divine_target(self, alive_agents: List[str], role_probs: Dict[str, Dict[str, float]], my_id: str, history: List[str]) -> str:
        """占い先を決定：未調査の生存者の中から、最も人狼確率が高い人を選択"""
        target = None
        max_wolf_prob = -1.0

        for agent_id in alive_agents:
            if agent_id == my_id or agent_id in history: continue
            
            prob_wolf = role_probs.get(agent_id, {}).get("WEREWOLF", 0.0)
            if prob_wolf > max_wolf_prob:
                max_wolf_prob = prob_wolf
                target = agent_id
        
        return target or alive_agents[0]

    def decide_attack_target(self, alive_agents: List[str], role_probs: Dict[str, Dict[str, float]], my_id: str) -> str:
        """襲撃先を決定：村人陣営の脅威（SEER, BODYGUARD）が高い順に選択"""
        target = None
        max_threat = -1.0

        for agent_id in alive_agents:
            if agent_id == my_id: continue
            
            p = role_probs.get(agent_id, {})
            threat = p.get("SEER", 0.0) + p.get("BODYGUARD", 0.0)
            
            if threat > max_threat:
                max_threat = threat
                target = agent_id
        
        return target or alive_agents[0]
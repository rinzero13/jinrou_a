from typing import List, Dict
from aiwolf_nlp_common.packet import Role

class ActionModule:
    """
    信念モデルの役職確率と対話要約を用いて、具体的な行動（投票・占い・襲撃）を決定する。
    """
    def __init__(self, logger):
        self.logger = logger

    def decide_vote_target(self, alive_agents: List[str], role_probs: Dict[str, Dict[str, float]], my_role: Role, my_id: str) -> str:
        """投票先を決定：自分以外の生存者から選択し、役職判定を安全に行う"""
        # 自分を除いた候補リストを作成
        candidates = [a for a in alive_agents if a != my_id]
        if not candidates:
            return str(my_id)

        # 【修正点】my_roleを文字列に変換して比較（Roleオブジェクトと文字列の混同によるエラーを防止）
        my_role_name = my_role.name if hasattr(my_role, 'name') else str(my_role)

        target = candidates[0]
        max_score = -1.0

        for agent_id in candidates:
            p = role_probs.get(agent_id, {})
            
            # 【修正点】文字列ベースで役職陣営を判定
            if my_role_name not in ["WEREWOLF", "POSSESSED"]:
                # 村人陣営視点：人狼（WEREWOLF）または狂人（POSSESSED）の合算確率が高い順
                score = p.get("WEREWOLF", 0.0) + p.get("POSSESSED", 0.0)
            else:
                # 人狼陣営視点：村人陣営の役職者（SEER, BODYGUARD）らしい順
                score = p.get("SEER", 0.0) + p.get("BODYGUARD", 0.0) + p.get("VILLAGER", 0.0)

            if score > max_score:
                max_score = score
                target = agent_id
        
        # 【修正点】決定プロセスをログに出力
        self.logger.logger.info(f"Decided vote target: {target} (MyRole: {my_role_name}, Score: {max_score})")
        
        # 【修正点】戻り値が確実に文字列であることを保証
        return str(target)

    def decide_divine_target(self, alive_agents: List[str], role_probs: Dict[str, Dict[str, float]], my_id: str, history: List[str]) -> str:
        """占い先を決定：自分以外、かつ未調査の生存者から選択"""
        candidates = [a for a in alive_agents if a != my_id and a not in history]
        
        if not candidates:
            # 占える相手がいない場合は自分以外の生存者から選ぶ（エラー回避）
            candidates = [a for a in alive_agents if a != my_id]
            if not candidates: return str(my_id)
            return str(candidates[0])

        target = candidates[0]
        max_wolf_prob = -1.0

        for agent_id in candidates:
            prob_wolf = role_probs.get(agent_id, {}).get("WEREWOLF", 0.0)
            if prob_wolf > max_wolf_prob:
                max_wolf_prob = prob_wolf
                target = agent_id
        
        self.logger.logger.info(f"Decided divine target: {target} (Wolf Prob: {max_wolf_prob})")
        return str(target)

    def decide_attack_target(self, alive_agents: List[str], role_probs: Dict[str, Dict[str, float]], my_id: str) -> str:
        """襲撃先を決定：自分以外の生存者から選択"""
        candidates = [a for a in alive_agents if a != my_id]
        if not candidates:
            return str(alive_agents[0] if alive_agents else my_id)

        target = candidates[0]
        max_threat = -1.0

        for agent_id in candidates:
            p = role_probs.get(agent_id, {})
            threat = p.get("SEER", 0.0) + p.get("BODYGUARD", 0.0)
            
            if threat > max_threat:
                max_threat = threat
                target = agent_id
        
        self.logger.logger.info(f"Decided attack target: {target} (Threat Score: {max_threat})")
        return str(target)
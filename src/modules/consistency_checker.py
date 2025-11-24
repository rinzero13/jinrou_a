# src/modules/consistency_checker.py

import json
from aiwolf_nlp_common.packet import Info, Talk
from utils.agent_logger import AgentLogger
from openai import OpenAI, APIError

LLM_MODEL = "gpt-3.5-turbo" # Agent.pyと同じモデルを使用

class LogicalConsistencyChecker:
    def __init__(self, logger: AgentLogger, openai_client: OpenAI | None):
        self.logger = logger
        self.openai_client = openai_client
        
    def _create_checker_prompt(self, game_info: Info, talk_history: list[Talk], virtual_talk: str, agent_name: str) -> tuple[str, str]:
        """矛盾判定用のLLMプロンプトを生成する。"""
        # ここでは簡略化のため、システムメッセージにゲーム情報を集約する
        
        # プレイヤー名（人名）の安全な取得
        agent_name_map = getattr(game_info, 'agent_name_map', {})
        agent_display_name = agent_name_map.get(agent_name, agent_name)
        
        # 評価対象の発話者名を取得
        speaker_name = agent_display_name
        
        # 確定情報のサマリーを構築
        executed_name = agent_name_map.get(game_info.executed_agent, "なし") if game_info.executed_agent else "なし"
        attacked_name = agent_name_map.get(game_info.attacked_agent, "なし") if game_info.attacked_agent else "なし"
        alive_agents_list = [agent_name_map.get(a, a) for a in game_info.status_map if game_info.status_map[a] == 'ALIVE']

        # 判定基準のシステムメッセージを構築 (ゼミ資料P.30-33を参照)
        system_message = (
            "あなたは人狼ゲームのAIの発話を分析する専門家です。\n"
            "以下の【現在のゲーム情報】、【発話履歴】、【評価対象の発話】に基づき、"
            "その発話が**論理的一貫性を保っているか**を判定してください。\n"
            "特に以下の4点に基づき、矛盾の有無を厳格に評価すること。論理的な飛躍も矛盾と見なします。\n"
            "1. **客観的事実・状況との関係** (ルール、既知の事実、可能性の範囲との矛盾)\n"
            "2. **発話者の行動・役職との関係** (過去の投票・CO・真の役職との矛盾)\n"
            "3. **他のプレイヤーの発話との関係** (応答性、主張の根拠、矛盾の指摘の妥当性)\n"
            "4. **ゲーム外の矛盾** (発話ルールに従っていない、不自然な発話)\n"
            "**発言者が人狼陣営の場合でも、発言自体に論理的な矛盾（例：占い結果と違うことを言っている、既知の事実を無視しているなど）があれば、論理的一貫性なしと判定してください。**\n"
        )
        
        user_prompt = (
            f"【現在のゲーム情報】\n"
            f"日目: {game_info.day}, 生存者: {', '.join(alive_agents_list)}\n"
            f"確定情報: 前回追放: {executed_name} / 前回襲撃: {attacked_name}\n"
            f"【発話履歴（直近5件）】\n"
            f"{json.dumps([{'agent': agent_name_map.get(t.agent, t.agent), 'day': t.day, 'text': t.text} for t in talk_history[-5:]], ensure_ascii=False, indent=2)}\n"
            f"【評価対象の発話】\n"
            f"発話者: {speaker_name}\n"
            f"発話内容: {virtual_talk}\n\n"
            f"**出力形式:**\n"
            f"必ず以下のJSON形式で結果を返してください。reasoningには、論理的矛盾がある場合の詳細な理由と、**修正するための具体的な指示**を記述してください。\n"
            f"{{ \"is_consistent\": true | false, \"reasoning\": \"評価の詳細な理由と修正指示を記述\" }}\n"
        )

        return system_message, user_prompt


    def check(self, game_info: Info, talk_history: list[Talk], virtual_talk: str, agent_name: str) -> tuple[bool, str]:
        """発話の論理的一貫性をLLMで判定する。"""
        if not self.openai_client:
            self.logger.logger.warning("OpenAI client not initialized. Skipping consistency check (M1).")
            return True, ""

        system_message, user_prompt = self._create_checker_prompt(game_info, talk_history, virtual_talk, agent_name)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.0 # 判定なので低く設定
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # JSONの解析
            try:
                result = json.loads(response_content)
                is_consistent = result.get("is_consistent", False)
                reason = result.get("reasoning", "詳細な理由が見つかりませんでした。")
                
                self.logger.logger.info(f"M1 Check Result: Consistent={is_consistent}, Reason={reason[:50]}...")
                return is_consistent, reason
                
            except json.JSONDecodeError:
                self.logger.logger.error(f"M1 Checker failed to parse JSON: {response_content[:100]}...")
                # 解析失敗時は安全のため矛盾なし(true)として扱う
                return True, "JSON解析エラーによりスキップされました。"
                
        except APIError as e:
            self.logger.logger.error("M1 Checker OpenAI API Error: %s", e)
            return True, "APIエラーによりスキップされました。"
        except Exception as e:
            self.logger.logger.error("M1 Checker unexpected error: %s", e)
            return True, "予期せぬエラーによりスキップされました。"
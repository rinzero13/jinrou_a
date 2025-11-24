"""Module that defines the base class for agents.

エージェントの基底クラスを定義するモジュール.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread

import os
from openai import OpenAI, APIError

# --- 新規モジュールのインポート ---
from modules.policy_generator import UtterancePolicyGenerator
from modules.lie_strategist import LieStrategyModule
from modules.consistency_checker import LogicalConsistencyChecker

LLM_MODEL = "gpt-3.5-turbo"

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """Base class for agents.

    エージェントの基底クラス.
    """

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """Initialize the agent.

        エージェントの初期化を行う.

        Args:
            config (dict[str, Any]): Configuration dictionary / 設定辞書
            name (str): Agent name / エージェント名
            game_id (str): Game ID / ゲームID
            role (Role): Role / 役職
        """
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        # --- OpenAIクライアントの初期化 ---
        self.openai_client: OpenAI | None = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = OpenAI()
            except Exception as e:
                # 環境変数が設定されていても初期化に失敗した場合
                self.agent_logger.logger.error("Failed to initialize OpenAI client: %s", e)
        else:
            # 環境変数が設定されていない場合、ログで警告
            self.agent_logger.logger.warning("OPENAI_API_KEY environment variable is not set. Using random talk mode.")
        # ---------------------------------------------
        
        # --- カスタムモジュールの設定と初期化 ---
        module_settings = self.config.get("custom_modules", {})
        
        self.USE_M3_POLICY = module_settings.get("enable_module_policy", False)
        self.USE_M2_LIE = module_settings.get("enable_module_lie", False)
        self.USE_M1_CONSISTENCY = module_settings.get("enable_module_consistency", False)
        self.MAX_REGENERATION_ATTEMPTS = module_settings.get("max_regeneration_attempts", 3)

        # モジュールインスタンスの作成
        self.M3_Policy = UtterancePolicyGenerator(self.agent_logger) if self.USE_M3_POLICY else None
        # M2はM3を前提とするが、ここでは独立したフラグで制御
        self.M2_Lie = LieStrategyModule(self.agent_logger) if self.USE_M2_LIE else None
        self.M1_Consistency = LogicalConsistencyChecker(self.agent_logger, self.openai_client) if self.USE_M1_CONSISTENCY and self.openai_client else None
        
        # ---------------------------------------------

        self.comments: list[str] = []
        with Path.open(
            Path(str(self.config["path"]["random_talk"])),
            encoding="utf-8",
        ) as f:
            self.comments = f.read().splitlines()

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """Decorator to set action timeout.

        アクションタイムアウトを設定するデコレータ.

        Args:
            func (Callable[P, T]): Function to be decorated / デコレート対象の関数

        Returns:
            Callable[P, T]: Function with timeout functionality / タイムアウト機能を追加した関数
        """

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """Set packet information.

        パケット情報をセットする.

        Args:
            packet (Packet): Received packet / 受信したパケット
        """
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        """Get the list of alive agents.

        生存しているエージェントのリストを取得する.

        Returns:
            list[str]: List of alive agent names / 生存エージェント名のリスト
        """
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def name(self) -> str:
        """Return response to name request.

        名前リクエストに対する応答を返す.

        Returns:
            str: Agent name / エージェント名
        """
        return self.agent_name

    def initialize(self) -> None:
        """Perform initialization for game start request.

        ゲーム開始リクエストに対する初期化処理を行う.
        """

    def daily_initialize(self) -> None:
        """Perform processing for daily initialization request.

        昼開始リクエストに対する処理を行う.
        """

    def whisper(self) -> str:
        """Return response to whisper request.

        囁きリクエストに対する応答を返す.

        Returns:
            str: Whisper message / 囁きメッセージ
        """
        return random.choice(self.comments)  # noqa: S311

    def talk(self) -> str:
        """Return response to talk request.

        トークリクエストに対する応答を返す.

        Returns:
            str: Talk message / 発言メッセージ
        """
        # --- OpenAI APIを呼び出すロジック ---
        if self.openai_client and self.info: # self.info (ゲーム情報) がセットされているか確認
            llm_response = self._call_openai_api()
            if llm_response:
                return llm_response
                
        # API呼び出しに失敗した場合やクライアントがない場合は、従来のランダム発言をフォールバックとして使用
        return random.choice(self.comments)  # noqa: S311

    def daily_finish(self) -> None:
        """Perform processing for daily finish request.

        昼終了リクエストに対する処理を行う.
        """

    def divine(self) -> str:
        """Return response to divine request.

        占いリクエストに対する応答を返す.

        Returns:
            str: Agent name to divine / 占い対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def guard(self) -> str:
        """Return response to guard request.

        護衛リクエストに対する応答を返す.

        Returns:
            str: Agent name to guard / 護衛対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def vote(self) -> str:
        """Return response to vote request.

        投票リクエストに対する応答を返す.

        Returns:
            str: Agent name to vote / 投票対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def attack(self) -> str:
        """Return response to attack request.

        襲撃リクエストに対する応答を返す.

        Returns:
            str: Agent name to attack / 襲撃対象のエージェント名
        """
        return random.choice(self.get_alive_agents())  # noqa: S311

    def finish(self) -> None:
        """Perform processing for game finish request.

        ゲーム終了リクエストに対する処理を行う.
        """

# src/agent/agent.py (Agentクラス内)

    def _create_talk_prompt(self, regeneration_feedback: str = None) -> tuple[str, str]:
            """
            LLMへのシステムメッセージとユーザープロンプトを生成する。
            M3とM2の指示を統合し、M1からの矛盾修正指示（regeneration_feedback）を組み込む。
            
            Args:
                regeneration_feedback (str): M1からの矛盾修正指示（ある場合、NoneでOK）
            
            Returns:
                tuple[str, str]: (system_message, user_prompt)
            """
            if not self.info:
                # ゲーム情報がない場合のフォールバック
                return "system_message", f"ゲーム情報がありません。ランダムな発言をします。エージェント名: {self.agent_name}" 

            # ----------------------------------------------------
            # 1. 基本情報の取得
            # ----------------------------------------------------
            agent_display_name = self._get_agent_display_name()
            
            # ----------------------------------------------------
            # 2. システムメッセージ (System Message) の構築
            # ----------------------------------------------------
            system_message = (
                f"あなたはAIWolf人狼ゲームのエージェント【{agent_display_name}】（ID：{self.agent_name}）です。あなたの役職は【{self.role.name}】です。\n"
                f"目的は役職の勝利条件を達成することです。\n"
                f"発言は**100文字以内**の簡潔な日本語で、あなたの役職が疑われないように行ってください。\n"
                f"**最終的な出力は、思考プロセスと発言内容を明確に区別し、必ず【発言：】というラベルを付けてください。**\n"
            )
            
            # --- M3: 発話方針決定・応答生成の指示の組み込み ---
            if self.USE_M3_POLICY and self.M3_Policy:
                system_message += self.M3_Policy.get_policy_instructions(self.role)
            
            # --- M2: 嘘戦略・意図決定の指示の組み込み ---
            if self.USE_M2_LIE and self.M2_Lie:
                system_message += self.M2_Lie.get_lie_strategy_instructions(self.role)
            
            # --- 役職ごとの具体的な戦略的指示（既存ロジックの継承）---
            match self.role:
                case Role.WEREWOLF:
                    system_message += "あなたは人狼です。正体を隠し、村人陣営の疑いをそらすことが最重要です。他のプレイヤーへの協力を装いつつ、村人同士の対立を煽り、議論を有利な方向へ誘導してください。特定のプレイヤーの発言の矛盾や信頼性を具体的に指摘しなさい。\n"
                case Role.POSSESSED:
                    system_message += "あなたは狂人です。人狼の勝利のために、村人陣営の議論を混乱させ、誤ったプレイヤー（村人陣営）に投票が集まるよう誘導してください。人狼の味方となる発言や、人狼を白く見せるような発言を心がけなさい。\n"
                case Role.SEER:
                    system_message += "あなたは占い師です。得られた情報（議論や占い結果）を基に、信頼性を高めつつ、村人陣営を勝利に導くために最も合理的なプレイヤーを指摘してください。議論をリードしなさい。\n"
                case _: # VILLAGER, BODYGUARD, MEDIUM, etc.
                    system_message += "あなたは村人陣営です。生存者の発言を注意深く分析し、論理的推論に基づき怪しいプレイヤーを指摘するか、他のプレイヤーと協力し、村人陣営の勝利を目指してください。\n"

            # --- M1: 論理矛盾修正指示の組み込み（再生成ループ時のみ） ---
            if regeneration_feedback:
                system_message += f"\n\n**【重要：論理矛盾の修正指示】**\n"
                system_message += f"あなたの前回の発言は以下の点で論理的に矛盾していると判定されました。**この修正指示を完全に反映し、矛盾を解消した発言を再生成してください。**\n"
                system_message += f"修正理由: {regeneration_feedback}\n"
                self.agent_logger.warning("M1 feedback integrated into system prompt for regeneration.")

            # ----------------------------------------------------
            # 3. ユーザープロンプト (User Prompt) の構築
            # ----------------------------------------------------
            
            game_state_summary = self._summarize_game_state()
            formatted_history = self._format_talk_history()
            cot_prompt = self._create_cot_prompt()

            user_prompt = f"""
                【現在のゲーム情報】
                {game_state_summary}

                【これまでの会話履歴（直近10件）】
                {formatted_history}

                【あなたの発言決定プロセス】
                あなたの役職の勝利のために、これらの情報に基づき、発言内容を決定してください。

                {cot_prompt}"""

            return system_message, user_prompt

    # ----------------------------------------------------------------------
    # Agent ID と Agent Name のマッピングに関するヘルパー関数
    # ----------------------------------------------------------------------

    def _get_agent_display_name(self) -> str:
            """現在のエージェントの表示名（シオン、ベンジャミンなど）を取得する。"""
            # Infoパケットの agent フィールド（表示名）を直接利用
            if self.info and hasattr(self.info, 'agent'):
                return self.info.agent
            # 取得できない場合は、初期化時のエージェント名（コネクタID）をフォールバックとして使用
            return self.agent_name

    def _get_role_knowledge(self) -> str:
        """役職固有の確定情報を返す。（修正不要だが再掲）"""
        # Role のインポートが必要 (ファイル上部に from aiwolf_nlp_common.packet import Role があるはず)
        if self.role == Role.WEREWOLF: return "相方人狼、狂人、襲撃対象"
        if self.role == Role.SEER: return "過去の占い結果"
        return "特になし"

    def _summarize_game_state(self) -> str:
        """現在のゲーム状況をサマリーする。（agent_list依存を解消）"""
        if not self.info: 
            return "ゲーム情報なし。"
        
        # status_map から生存エージェントの名前を取得
        alive_agent_names = [k for k, v in self.info.status_map.items() if v == 'ALIVE']
        
        # executed_agent や attacked_agent は Info オブジェクトに Agent Name (シズエなど) 
        # または Agent ID (kanolabX) で格納されているため、そのまま表示します。
        executed = self.info.executed_agent if self.info.executed_agent else "なし"
        attacked = self.info.attacked_agent if self.info.attacked_agent else "なし"
        
        return (
            f"日目: Day {self.info.day}\n"
            f"生存者: {', '.join(alive_agent_names)}\n"
            f"前回追放: {executed}\n"
            f"前回襲撃: {attacked}\n"
            f"あなたの役職の知っていること: {self._get_role_knowledge()}"
        )

    def _format_talk_history(self) -> str:
        """会話履歴を整形する。（Agent ID/Nameマッピングに依存しない形に修正）"""
        formatted = []
        # talk.agent には Agent Name (シズエなど) が格納されていると推測されるため、そのまま使用
        for talk in self.talk_history[-10:]:
            speaker = talk.agent 
            formatted.append(f"D{talk.day} {speaker}: {talk.text}")
        return "\n".join(formatted) if formatted else "まだ会話はありません。"
        
    def _create_cot_prompt(self) -> str:
        """CoTプロンプトを返す。（修正不要だが再掲）"""
        return (
            "**思考プロセス (Chain of Thought):**\n"
            "1. 現在の状況分析（勝敗条件と役職）:...\n"
            "2. 最適な次の行動（投票・CO・護衛・襲撃など）:...\n"
            "3. (M3) 主張の核と応答の決定:...\n"
            "4. (M2) 嘘の使用の有無と論理の構築:...\n"
            "5. 最終的な発言内容:...\n"
            "（この思考プロセスは発言には含めず、最終的な「発言：」のみを続けて出力すること）"
        )
    # ---------------------------------------------

    def _call_openai_api(self) -> str | None:
        """OpenAI APIを呼び出し、モジュールパイプラインに従って応答を取得・解析する。"""
        if not self.openai_client:
            return None

        # M1: 論理的一貫性判定・修正のためのループ
        feedback = None
        for attempt in range(self.MAX_REGENERATION_ATTEMPTS):
            # 1. LLMへのプロンプト生成 (M3/M2の指示を統合、またはM1のフィードバックを組み込む)
            system_message, user_prompt = self._create_talk_prompt(regeneration_feedback=feedback)
            
            try:
                # 2. LLM APIコール (仮想発話の生成)
                # ... (既存のAPIコールコード) ...
                response = self.openai_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=350,
                    temperature=0.7,
                )
                
                talk_content_full = response.choices[0].message.content.strip()
                
                # 3. 応答の解析と抽出（「発言：」のラベルを基に）
                # ... (既存の応答解析コード) ...
                
                if "発言：" in talk_content_full:
                    parts = talk_content_full.split("発言：", 1)
                    strategy_log = parts[0].strip()
                    talk_content = parts[1].strip()
                    
                    if len(talk_content) > 100:
                        talk_content = talk_content[:100]

                    self.agent_logger.logger.info(f"Attempt {attempt+1}: LLM Strategy (Internal COG): {strategy_log[:50]}...")
                    self.agent_logger.logger.info(f"Attempt {attempt+1}: LLM Response (Virtual Talk): {talk_content}")

                    # 4. M1: 論理的一貫性チェックの実行
                    # モジュールが有効化されている場合、チェックを行う
                    if self.USE_M1_CONSISTENCY and self.M1_Consistency:
                        is_consistent, reason = self.M1_Consistency.check(
                            game_info=self.info, 
                            talk_history=self.talk_history, 
                            virtual_talk=talk_content, 
                            agent_name=self.agent_name
                        )
                        
                        if is_consistent:
                            # 矛盾なしと判定されたら、ループを終了して発言を返す
                            self.agent_logger.logger.info("M1: Logical consistency check passed. Talk decided.")
                            return talk_content
                        else:
                            # 矛盾ありと判定されたら、feedbackを更新し、再生成へ
                            self.agent_logger.logger.warning(f"M1: Logical inconsistency detected (Attempt {attempt+1}/{self.MAX_REGENERATION_ATTEMPTS}). Retrying...")
                            feedback = reason # M1からの修正指示をフィードバックとして設定
                            continue # 次のループへ
                    else:
                        # M1が無効の場合、チェックせずに発言を返す
                        self.agent_logger.logger.info("M1: Consistency check skipped (Module disabled). Talk decided.")
                        return talk_content

                # テンプレートに従わなかった場合 (既存コードのフォールバック)
                else:
                    self.agent_logger.logger.warning("LLM response did not contain the '発言：' tag. Using full content as talk.")
                    talk_content = talk_content_full[:100] if len(talk_content_full) > 100 else talk_content_full
                    
                    # テンプレート外でもM1はチェックすべき
                    if self.USE_M1_CONSISTENCY and self.M1_Consistency:
                         is_consistent, reason = self.M1_Consistency.check(
                            game_info=self.info, 
                            talk_history=self.talk_history, 
                            virtual_talk=talk_content, 
                            agent_name=self.agent_name
                        )
                         if is_consistent:
                            return talk_content
                         else:
                            feedback = reason
                            continue
                    
                    return talk_content # M1が無効、または最初の試行で矛盾なしの場合

            except APIError as e:
                self.agent_logger.logger.error("OpenAI API Error: %s", e)
                break 
            except Exception as e:
                self.agent_logger.logger.error("An unexpected error occurred during API call: %s", e)
                break

        # ループを抜けたが有効な発言が得られなかった場合
        self.agent_logger.logger.warning("Failed to generate consistent talk after all attempts. Using random talk fallback.")
        return random.choice(self.comments)
    

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """Execute action according to request type.

        リクエストの種類に応じたアクションを実行する.

        Returns:
            str | None: Action result string or None / アクションの結果文字列またはNone
        """
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None
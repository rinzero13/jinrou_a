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

# src/agent/agent.py (Agentクラス内)

# src/agent/agent.py (Agentクラス内)

# src/agent/agent.py (Agentクラス内)

    def _create_talk_prompt(self) -> tuple[str, str]:
        """LLMへのシステムメッセージとユーザープロンプトを生成する."""

        if not self.info:
            # TALKリクエスト時に通常は発生しないが、安全のためにチェック
            return "system_message", f"ゲーム情報がありません。ランダムな発言をします。エージェント名: {self.agent_name}" 

        # --- 1. プレイヤー名（人名）の安全な取得とマップの準備 ---
        # self.info.agent_name_map が存在しない場合に備えて空の辞書をフォールバックとして設定
        agent_name_map = getattr(self.info, 'agent_name_map', {})
        
        # 自分のAgent IDに対応する人名を取得。存在しない場合はAgent IDをそのまま使用。
        agent_display_name = agent_name_map.get(self.agent_name, self.agent_name)
        # --------------------------------------------

        # 役職に基づいたシステムメッセージの構築
        system_message = (
            f"あなたはAIWolf人狼ゲームのエージェント【{agent_display_name}】（ID：{self.agent_name}）です。あなたの役職は【{self.role.name}】です。\n"
            f"目的は役職の勝利条件を達成することです。\n"
            f"発言は**100文字以内**の簡潔な日本語で、あなたの役職が疑われないように行ってください。\n"
        )
        
        # 役職ごとの具体的な戦略的指示 (資料 の知見を組み込み)
        match self.role:
            case Role.WEREWOLF:
                # 人狼の戦略：欺瞞と混乱の煽動
                system_message += "あなたは人狼です。正体を隠し、村人陣営の疑いをそらすことが最重要です。他のプレイヤーへの協力を装いつつ、村人同士の対立を煽り、議論を有利な方向へ誘導してください。特定のプレイヤーの発言の矛盾や信頼性を具体的に指摘しなさい。\n"
            case Role.POSSESSED:
                # 狂人の戦略：人狼の勝利を最優先し、村人陣営の議論を妨害
                system_message += "あなたは狂人です。人狼の勝利のために、村人陣営の議論を混乱させ、誤ったプレイヤー（村人陣営）に投票が集まるよう誘導してください。人狼の味方となる発言や、人狼を白く見せるような発言を心がけなさい。\n"
            case Role.SEER:
                # 占い師の場合、占い結果があればそれを強調
                divine_result = self.info.divine_result
                if divine_result:
                    target_name = agent_name_map.get(divine_result.target, divine_result.target)
                    # Note: divine_result.result はSpecies型または文字列なので、Role.WEREWOLFと比較
                    result_text = "人狼（黒）" if divine_result.result == Role.WEREWOLF else "人間（白）"
                    system_message += f"あなたは占い師です。Agent【{target_name}】を占った結果、【{result_text}】でした。この確定情報を基に、信頼性を高めつつ、村人陣営を勝利に導くために最も合理的なプレイヤーを指摘してください。無意味な発言や曖昧な発言は避け、議論をリードしなさい。\n"
                else:
                    system_message += "あなたは占い師です。得られた情報（議論）を基に、信頼性を高めつつ、村人陣営を勝利に導くために最も合理的なプレイヤーを指摘してください。議論をリードしなさい。\n"
            case _: # VILLAGER, BODYGUARD, MEDIUM, etc.
                # 村人陣営の戦略：論理的な推理と協調
                system_message += "あなたは村人陣営です。生存者の発言を注意深く分析し、論理的推論に基づき怪しいプレイヤーを指摘するか、他のプレイヤーと協力し、村人陣営の勝利を目指してください。\n"

        # 確定情報のサマリーを人名（またはID）で表示
        executed_id = self.info.executed_agent
        attacked_id = self.info.attacked_agent
        
        # IDを人名に変換 (get()を使用して安全にアクセス)
        executed_name = agent_name_map.get(executed_id, executed_id) if executed_id else "なし"
        attacked_name = agent_name_map.get(attacked_id, attacked_id) if attacked_id else "なし"

        # 生存エージェントのリストを人名（またはID）で作成
        alive_agents_list = [
            agent_name_map.get(agent_id, agent_id)
            for agent_id in self.get_alive_agents()
        ]

        # 会話履歴の取得（直近の5件に限定し、発言者も人名に変換）
        talk_history_summary = "\n".join(
            [f"【{agent_name_map.get(t.agent, t.agent)}】(Day{t.day}): {t.text[:40]}..." for t in self.talk_history[-5:]] 
        )
        
        user_prompt = (
            f"【現在のゲーム情報】\n"
            f"日目: {self.info.day}, 生存者: {', '.join(alive_agents_list)}\n"
            f"確定情報: 前回追放: {executed_name} / 前回襲撃: {attacked_name}\n"
            f"【あなたの目標】\n"
            f"あなたの役職【{self.role.name}】の勝利に貢献する発言をしてください。特に、誰の言動が矛盾しているか、誰が信頼できるかを具体的に言及してください。\n"
            f"【直近の会話履歴（参考）】\n"
            f"{talk_history_summary if talk_history_summary else 'まだ会話はありません。'}\n\n"
            f"発言:"
        )
        
        return system_message, user_prompt

    def _call_openai_api(self) -> str | None:
        """OpenAI APIを呼び出して応答を取得する."""
        if not self.openai_client:
            return None

        try:
            system_message, user_prompt = self._create_talk_prompt()
            
            response = self.openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200, 
                temperature=0.7, # 創造性の調整（0.0で確定的、1.0で多様）
            )
            
            talk_content = response.choices[0].message.content.strip()
            self.agent_logger.logger.info("LLM Response: %s", talk_content)
            return talk_content
            
        except APIError as e:
            self.agent_logger.logger.error("OpenAI API Error: %s", e)
            return None
        except Exception as e:
            self.agent_logger.logger.error("An unexpected error occurred during API call: %s", e)
            return None
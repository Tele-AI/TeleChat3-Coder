import vllm
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    BaseMultiModalItemTracker,
    _ChatTemplateContentFormat,
    _parse_chat_message_content,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _telechat3_parse_chat_message_content(
    message: ChatCompletionMessageParam,
    mm_tracker: BaseMultiModalItemTracker,
    content_format: _ChatTemplateContentFormat,
) -> list[ConversationMessage]:
    result = _parse_chat_message_content(message, mm_tracker, content_format)
    reasoning_content = message.get("reasoning_content")

    if len(result) > 0 and reasoning_content:
        logger.info("add reasoning content to input prompt.")
        result[0].update({"reasoning_content": reasoning_content})

    return result


def register_reasoning():
    vllm.entrypoints.chat_utils._parse_chat_message_content = (
        _telechat3_parse_chat_message_content
    )

import ast
import json
import regex as re
from collections.abc import Sequence
from typing import List, Any

from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)

from vllm.logger import init_logger

logger = init_logger(__name__)


def _is_string_type(
    tool_name: str, arg_name: str, tools: List[ChatCompletionToolsParam] | None
):
    if tools is None:
        return False
    for tool in tools:
        if tool.function.name == tool_name:
            if tool.function.parameters is None:
                return False
            arg_type = (
                tool.function.parameters.get("properties", {})
                .get(arg_name, {})
                .get("type", None)
            )
            return arg_type == "string"
    logger.debug("No tool named '%s'.", tool_name)
    return False


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


@ToolParserManager.register_module("telechat3")
class TeleChat3ModelToolParser(ToolParser):
    """
    Tool call parser for TeleChat3-36B models.
    Used when --enable-auto-tool-choice --tool-call-parser telechat3
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        super().__init__(tokenizer)

        # initialize properties used for state when parsing tool calls in
        # streaming mode
        self.current_tool_id: int = -1

        self.tool_start_token = "<tool_call>"
        self.tool_end_token = "</tool_call>"

        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<param_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<param_key>(.*?)</param_key>(?:\\n|\s)*<param_value>(.*?)</param_value>",
            re.DOTALL,
        )
        self._buffer = ""

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest):

        matched_tool_calls = self.func_detail_regex.findall(model_output)
        logger.debug("model_output: %s", model_output)

        tool_calls = []
        try:
            for match in matched_tool_calls:
                tc_name = match[0].strip()
                arg_dict = {}
                if len(match) > 1:
                    for key, value in self.func_arg_regex.findall(match[1]):
                        arg_key = key.strip()
                        arg_val = value.strip()
                        if not _is_string_type(tc_name, key, request.tools):
                            arg_val = _deserialize(arg_val)
                        logger.debug("arg_key = %s, arg_val = %s", arg_key, arg_val)
                        arg_dict[arg_key] = arg_val
                tool_calls.append(
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tc_name,
                            arguments=json.dumps(arg_dict, ensure_ascii=False),
                        ),
                    )
                )
        except Exception:
            logger.exception("Failed to extract tool call spec")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        else:
            if len(tool_calls) > 0:
                content = model_output[: model_output.find(self.tool_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True, tool_calls=tool_calls, content=content
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        self._buffer += delta_text
        cur_text = self._buffer
        start_idx = cur_text.find(self.tool_start_token)
        if start_idx == -1:
            self._buffer = ""
            return DeltaMessage(content=cur_text)
        logger.debug("cur_text = %s", cur_text)
        end_idx = cur_text.find(self.tool_end_token)
        if end_idx != -1:
            extracted_tool_calls = self.extract_tool_calls(
                cur_text[: end_idx + len(self.tool_end_token)], request
            )
            if len(extracted_tool_calls.tool_calls) == 0:
                logger.warning("Failed to extract any tool calls.")
                return None
            self.current_tool_id += 1
            tool_call = extracted_tool_calls.tool_calls[0]
            delta = DeltaMessage(
                content=extracted_tool_calls.content,
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_id,
                        id=tool_call.id,
                        type=tool_call.type,
                        function=DeltaFunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                    )
                ],
            )
            self._buffer = cur_text[end_idx + len(self.tool_end_token) :]
            return delta
        self._buffer = cur_text[start_idx:]
        return DeltaMessage(content=cur_text[:start_idx])


def register_tool_parser(): ...

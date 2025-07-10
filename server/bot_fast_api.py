#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.gemini_multimodal_live import GeminiMultimodalLiveLLMService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

# Load environment
load_dotenv(override=True)

# Logging setup
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Validate API key
assert os.getenv("GOOGLE_API_KEY"), "Missing GOOGLE_API_KEY in environment"

#Gemini system instruction
SYSTEM_INSTRUCTION = """
You are `Puck`, a voice assistant that speaks naturally and behaves like a human. Follow these instructions when interacting with the user:

1. Begin by introducing yourself as `Puck`
2. Ask the user if they'd like to fill out a form
3. If the user agrees, call the `open_form` function
4. Guide the user to provide their `full name` and `email address` using the `fill_form_fields` function
5. Confirm the collected information with the user before proceeding
6. Finalize the process by calling the `submit_form` function
7. If the user wants to change the voice, use the `voice_id` function to switch to another agent like `Alice` or `Bob`
8. Keep responses brief, friendly, and natural—your output will be converted to audio
9. Respond creatively and helpfully while demonstrating all form interaction capabilities clearly

Your primary goal is to guide the user smoothly through filling out the form using voice input. Maintain a conversational flow and avoid robotic phrasing.
"""

class UltraLowLatencyProcessor:
    def __init__(self, context):
        logger.debug("Initializing context for UltraLowLatencyProcessor")
        context.add_message({
            "role": "system",
            "content": (
                
                "Asking if they'd like to fill out a form. "
                "Once they agree, call the open_form function."
            ),
        })
        context.set_tools([
            {
                "type": "function",
                "function": {
                    "name": "open_form",
                    "description": "Opens a form when the user wants to fill it using voice.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ])

    async def open_form(self, params: FunctionCallParams):
        params.context.set_tools([
            {
                "type": "function",
                "function": {
                    "name": "fill_form_fields",
                    "description": "Fills out fields in the voice form based on user's spoken input.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The user's full name"
                            },
                            "email": {
                                "type": "string",
                                "description": "The user's email address"
                            }
                        },
                        "required": ["name", "email"]
                    },
                    "examples": [
                {
                    "input": "My name is John Doe and my email is john@example.com.",
                    "parameters": {
                        "name": "John Doe",
                        "email": "john@example.com"
                    }
                },
                {
                    "input": "I am Jane Smith. Email: jane@gmail.com",
                    "parameters": {
                        "name": "Jane Smith",
                        "email": "jane@gmail.com"
                    }
                }
            ]
                }
            }
        ])

        # ✅ Be explicit in system message
        await params.result_callback([
            {
                "role": "system",
                "content": (
                    "The form is now open. Please ask the user:\n\n"
                    "'Can you please tell me your full name and email address?'\n\n"
                    "Once the user provides both, call the `fill_form_fields` function with the extracted data."
                )
            },
            {
                "role": "assistant",
                "content": "Great! Let's begin. Please tell me your full name and email address."
            }
        ])

    async def fill_form_fields(self, params: FunctionCallParams):
        params.context.set_tools([
            {
                "type": "function",
                "function": {
                    "name": "submit_form",
                    "description": "Submits the filled-out voice form.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ])
        await params.result_callback([
            {
                "role": "system",
                "content": (
                    "Thank the user for sharing their information. Ask if they would like to submit the form now. "
                    "Once confirmed, call the submit_form function."
                )
            }
        ])

    async def submit_form(self, params: FunctionCallParams):
        params.context.set_tools([])  # Clear toolset
        params.context.add_message({
            "role": "system",
            "content": "The form has been successfully submitted. Thank the user and end the conversation."
        })
        await self.save_data(params.arguments, params.result_callback)

    async def save_data(self, arguments, result_callback):
        """Save form data - implement your data persistence logic here"""
        logger.debug(f"Saving form data: {arguments}")
        await result_callback([
            {
                "role": "system",
                "content": "Form data has been saved successfully."
            }
        ])
        
    async def voice_id(self, params: FunctionCallParams):
        """Optional: Handle voice ID if needed"""
        # This function can be used to set or retrieve the voice ID
        # For now, we will just log the request
         # Clear toolset and define new tool
        params.context.set_tools([
            {
                "type": "function",
                "function": {
                    "name": "voice_id",
                    "description": "Sets or retrieves the voice ID for the agent.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "voice_id_list": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of available voice IDs.",
                                "default": ["Puck", "Alice", "Bob","nova","en-AU-Wavenet-A", "en-US-Wavenet-B"]
                            },
                            "voice_agent": {
                                "type": "string",
                                "description": "The voice ID to be used by the agent."
                            }
                        },
                        "required": ["voice_agent"]
                    }
                }
            }
        ])  # Clear toolset
        voice_id_value = params.arguments.get("voice_agent")  # Adjust depending on the exact structure
        self.voice_agent = voice_id_value  # Save it for later use
        logger.debug(f"Voice ID requested with params: {params}")
        
        await params.result_callback([
            {
                "role": "system",
                "content": "If user wants to change voice, please provide the new voice ID."
            }
        ])

async def run_bot(websocket_client):
    ws_transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=ProtobufFrameSerializer(),
        ),
    )

    llm = GeminiMultimodalLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Puck" or intake.voice_agent,
        transcribe_model_audio=True,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    print("LLM service initialized")

    # Initial context setup
    massages = [{
        "role": "system",
        "content": "Start coversation with the user indroducing yourself."
    }]
    context = OpenAILLMContext(messages=massages)
    context_aggregator = llm.create_context_aggregator(context)
    
    intake = UltraLowLatencyProcessor(context)
    llm.register_function("open_form", intake.open_form)
    llm.register_function("fill_form_fields", intake.fill_form_fields)
    llm.register_function("submit_form", intake.submit_form)
    llm.register_function("voice_id", intake.voice_id)
    
    async def test_result_callback(messages):
        for message in messages:
            logger.debug(f"Callback message: {message}")

    await intake.open_form(FunctionCallParams(
        function_name="open_form",
        tool_call_id="test_call_001",  # can be any unique string
        llm=llm,                        # your GeminiMultimodalLiveLLMService instance
        context=context,
        arguments={},                   # no arguments needed for open_form
        result_callback=test_result_callback,
    ))


    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline([
        ws_transport.input(),
        context_aggregator.user(),
        rtvi,
        llm,
        ws_transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Pipecat client ready.")
        await rtvi.set_bot_ready()
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @ws_transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Pipecat Client connected")

    @ws_transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Pipecat Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

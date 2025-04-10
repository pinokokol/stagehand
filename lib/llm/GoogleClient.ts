import {
  GoogleGenerativeAI,
  GenerationConfig,
  Content,
  Part,
  HarmBlockThreshold,
  HarmCategory,
} from "@google/generative-ai";
import {
  LLMClient,
  CreateChatCompletionOptions,
  ChatMessage,
  LLMResponse,
  ChatMessageImageContent,
} from "./LLMClient";
import { AvailableModel, ClientOptions } from "../../types/model";
import { StagehandError } from "@/types/stagehandErrors";
import { LogLine } from "@/types/log";

// Helper function to map project's ChatMessage format to Google's Content format
const mapMessagesToGoogleContent = (messages: ChatMessage[]): Content[] => {
  return messages.map((msg) => {
    const parts: Part[] = [];

    if (typeof msg.content === "string") {
      parts.push({ text: msg.content });
    } else {
      // Handle complex content (images, text parts)
      msg.content.forEach((part) => {
        if ("text" in part && part.text) {
          parts.push({ text: part.text });
        } else if ("image_url" in part && part.image_url) {
          // Handle image URL
          if (part.image_url.url.startsWith("data:")) {
            // For base64 data URLs
            const match = part.image_url.url.match(
              /^data:image\/([a-zA-Z]+);base64,(.*)$/,
            );
            if (match) {
              const mimeType = `image/${match[1]}`;
              const base64Data = match[2];
              parts.push({
                inlineData: {
                  mimeType: mimeType,
                  data: base64Data,
                },
              });
            }
          } else {
            // For regular URLs
            parts.push({
              fileData: {
                mimeType: "image/jpeg", // Default to JPEG if unknown
                fileUri: part.image_url.url,
              },
            });
          }
        } else if (
          "source" in part &&
          part.source &&
          typeof part.source === "object" &&
          "type" in part.source &&
          part.source.type === "image" &&
          "data" in part.source
        ) {
          // Handle source data with proper type checking
          parts.push({
            inlineData: {
              mimeType:
                "media_type" in part.source
                  ? part.source.media_type
                  : "image/jpeg",
              data: part.source.data,
            },
          });
        }
      });
    }

    // Google uses 'user' and 'model' roles
    const role = msg.role === "assistant" ? "model" : "user";

    return { role, parts };
  });
};

// Helper to prepare content for Google API - consolidate system messages and ensure proper role alternation
const prepareGoogleContent = (
  contents: Content[],
  systemInstruction?: string,
): Content[] => {
  if (!contents || contents.length === 0) {
    return systemInstruction
      ? [{ role: "user", parts: [{ text: systemInstruction }] }]
      : [];
  }

  // Start with system instruction if provided
  const result: Content[] = [];

  // Add system instruction to first user message if present
  if (systemInstruction && contents[0].role === "user") {
    const firstUserMessage = { ...contents[0] };
    firstUserMessage.parts = [
      { text: systemInstruction + "\n\n" },
      ...firstUserMessage.parts,
    ];
    result.push(firstUserMessage);
    contents = contents.slice(1);
  } else if (systemInstruction) {
    // Add system instruction as a separate message
    result.push({ role: "user", parts: [{ text: systemInstruction }] });
  }

  // Process the rest of the messages
  for (const content of contents) {
    // Skip if we just added the first message with system instruction
    if (
      result.length === 0 ||
      content.role !== result[result.length - 1].role
    ) {
      result.push(content);
    } else {
      // Merge consecutive messages with the same role
      const lastIdx = result.length - 1;
      result[lastIdx].parts = [...result[lastIdx].parts, ...content.parts];
    }
  }

  return result;
};

class GoogleClient extends LLMClient {
  public type = "google" as const;
  public hasVision = true;
  private client: GoogleGenerativeAI;
  private safetySettings: {
    category: HarmCategory;
    threshold: HarmBlockThreshold;
  }[];

  constructor({
    modelName,
    clientOptions,
    userProvidedInstructions,
  }: {
    logger: (message: LogLine) => void;
    modelName: AvailableModel;
    clientOptions?: ClientOptions;
    userProvidedInstructions?: string;
  }) {
    super(modelName, userProvidedInstructions);

    const apiKey = clientOptions?.apiKey;
    if (!apiKey) {
      throw new StagehandError(
        "Google API key is required. Pass it via GOOGLE_API_KEY environment variable or directly in Stagehand config.",
      );
    }

    this.client = new GoogleGenerativeAI(apiKey);
    this.modelName = modelName;
    this.clientOptions = clientOptions ?? {};

    // Default to permissive safety settings
    this.safetySettings = [
      {
        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
      },
      {
        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
      },
      {
        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
      },
      {
        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
      },
    ];
  }

  async createChatCompletion<T = LLMResponse>({
    options,
    logger,
  }: CreateChatCompletionOptions): Promise<T> {
    const {
      messages,
      temperature,
      top_p,
      maxTokens,
      image,
      requestId,
      response_model,
    } = options;

    logger({
      category: "google",
      message: "creating chat completion",
      level: 2,
      auxiliary: {
        options: {
          value: JSON.stringify({
            ...options,
            image: image ? "image-provided" : undefined,
          }),
          type: "object",
        },
        modelName: { value: this.modelName, type: "string" },
        requestId: { value: requestId, type: "string" },
      },
    });

    const generativeModel = this.client.getGenerativeModel({
      model: this.modelName,
      safetySettings: this.safetySettings,
    });

    const generationConfig: GenerationConfig = {
      ...(temperature !== undefined && { temperature }),
      ...(top_p !== undefined && { topP: top_p }),
      ...(maxTokens !== undefined && { maxOutputTokens: maxTokens }),
    };

    // Handle image if provided but not in messages
    const messagesWithImage = [...messages];
    if (image && image.buffer) {
      // Find the last user message to add the image to
      const lastUserMsgIndex = messagesWithImage
        .map((msg, i) => ({ role: msg.role, index: i }))
        .filter((item) => item.role === "user")
        .pop()?.index;

      if (lastUserMsgIndex !== undefined) {
        const base64Image = image.buffer.toString("base64");
        const imgContent: ChatMessageImageContent = {
          type: "image",
          image_url: { url: `data:image/jpeg;base64,${base64Image}` },
        };

        // Add image to the last user message
        const lastUserMsg = messagesWithImage[lastUserMsgIndex];
        if (typeof lastUserMsg.content === "string") {
          messagesWithImage[lastUserMsgIndex] = {
            ...lastUserMsg,
            content: [{ type: "text", text: lastUserMsg.content }, imgContent],
          };
        } else {
          messagesWithImage[lastUserMsgIndex] = {
            ...lastUserMsg,
            content: [...lastUserMsg.content, imgContent],
          };
        }
      }
    }

    // Map and prepare messages for Google API
    let googleContents = mapMessagesToGoogleContent(messagesWithImage);
    googleContents = prepareGoogleContent(
      googleContents,
      this.userProvidedInstructions,
    );

    try {
      const result = await generativeModel.generateContent({
        contents: googleContents,
        generationConfig,
      });

      const response = result.response;
      const candidate = response.candidates?.[0];

      if (!candidate?.content?.parts?.[0]) {
        const blockReason = response.promptFeedback?.blockReason;
        const finishReason = candidate?.finishReason;
        const safetyRatings = candidate?.safetyRatings
          ? JSON.stringify(candidate.safetyRatings)
          : "N/A";

        logger({
          category: "google",
          message: `No content received. FinishReason: ${finishReason}, BlockReason: ${blockReason}, SafetyRatings: ${safetyRatings}`,
          level: 0,
          auxiliary: {
            response: { value: JSON.stringify(response), type: "object" },
          },
        });

        throw new StagehandError(
          `Google API call failed or returned no content. FinishReason: ${finishReason}, BlockReason: ${blockReason}`,
        );
      }

      // Extract content from the response
      let responseText = "";
      for (const part of candidate.content.parts) {
        if ("text" in part && part.text) {
          responseText += part.text;
        }
      }

      // Usage data
      const usageData = {
        prompt_tokens: response.usageMetadata?.promptTokenCount || 0,
        completion_tokens: response.usageMetadata?.candidatesTokenCount || 0,
        total_tokens: response.usageMetadata?.totalTokenCount || 0,
      };

      // If a response model is provided, structure the response differently
      if (response_model) {
        try {
          // Try to parse the response text as JSON
          const parsedData = JSON.parse(responseText);
          const finalParsedResponse = {
            data: parsedData,
            usage: usageData,
          } as unknown as T;

          logger({
            category: "google",
            message: "chat completion with schema successful",
            level: 1,
            auxiliary: {
              usage: {
                value: JSON.stringify(usageData),
                type: "object",
              },
            },
          });

          return finalParsedResponse;
        } catch (error) {
          // Fallback for non-parseable JSON: format as the completion itself
          const finalParsedResponse = {
            data: {
              completed: responseText,
              progress: 1.0, // Assuming complete progress
            },
            usage: usageData,
          } as unknown as T;

          logger({
            category: "google",
            message:
              "chat completion with schema successful (using text fallback)",
            level: 1,
            auxiliary: {
              usage: {
                value: JSON.stringify(usageData),
                type: "object",
              },
              error: {
                value: `JSON parsing error: ${error.message}`,
                type: "string",
              },
            },
          });

          return finalParsedResponse;
        }
      }

      // Standard response format for non-schema requests
      const mappedResponse: LLMResponse = {
        id: response.usageMetadata?.promptTokenCount?.toString() ?? requestId,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: this.modelName,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: responseText,
              tool_calls: [], // Google doesn't support tool calls in the same way
            },
            finish_reason: candidate.finishReason || "stop",
          },
        ],
        usage: usageData,
      };

      logger({
        category: "google",
        message: "chat completion successful",
        level: 1,
        auxiliary: {
          usage: {
            value: JSON.stringify(mappedResponse.usage),
            type: "object",
          },
        },
      });

      return mappedResponse as T;
    } catch (error) {
      logger({
        category: "google",
        message: `Error creating chat completion: ${error.message}`,
        level: 0,
        auxiliary: {
          error: {
            value: JSON.stringify({
              message: error.message,
              name: error.name,
              stack: error.stack,
            }),
            type: "object",
          },
        },
      });

      throw new StagehandError(`Google API Error: ${error.message}`);
    }
  }
}

export default GoogleClient;

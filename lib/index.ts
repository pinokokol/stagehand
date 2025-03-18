import { Browserbase } from "@browserbasehq/sdk";
import { randomUUID } from "crypto";
import dotenv from "dotenv";
import fs from "fs";
import os from "os";
import path from "path";
import { z } from "zod";
import { BrowserResult } from "../types/browser";
import { EnhancedContext } from "../types/context";
import { LogLine } from "../types/log";
import { AvailableModel } from "../types/model";
import { Page } from "../types/page";
import {
  ActOptions,
  ActResult,
  ConstructorParams,
  ExtractOptions,
  ExtractResult,
  InitOptions,
  InitResult,
  LocalBrowserLaunchOptions,
  ObserveOptions,
  ObserveResult,
  AgentConfig,
  StagehandMetrics,
  StagehandFunctionName,
} from "../types/stagehand";
import { StagehandContext } from "./StagehandContext";
import { StagehandPage } from "./StagehandPage";
import { StagehandAPI } from "./api";
import { LLMClient } from "./llm/LLMClient";
import { LLMProvider } from "./llm/LLMProvider";
import { logLineToString } from "./utils";
import { ApiResponse, ErrorResponse } from "@/types/api";
import { AgentExecuteOptions, AgentResult } from "../types/agent";
import { StagehandAgentHandler } from "./handlers/agentHandler";
import puppeteer from "puppeteer-core";

dotenv.config({ path: ".env" });

const DEFAULT_MODEL_NAME = "gpt-4o";
const BROWSERBASE_REGION_DOMAIN = {
  "us-west-2": "wss://connect.usw2.browserbase.com",
  "us-east-1": "wss://connect.use1.browserbase.com",
  "eu-central-1": "wss://connect.euc1.browserbase.com",
  "ap-southeast-1": "wss://connect.apse1.browserbase.com",
};

async function getBrowser(
  apiKey: string | undefined,
  projectId: string | undefined,
  env: "LOCAL" | "BROWSERBASE" = "LOCAL",
  headless: boolean = false,
  logger: (message: LogLine) => void,
  browserbaseSessionCreateParams?: Browserbase.Sessions.SessionCreateParams,
  browserbaseSessionID?: string,
  localBrowserLaunchOptions?: LocalBrowserLaunchOptions,
): Promise<BrowserResult> {
  if (env === "BROWSERBASE") {
    if (!apiKey) {
      logger({
        category: "init",
        message:
          "BROWSERBASE_API_KEY is required to use BROWSERBASE env. Defaulting to LOCAL.",
        level: 0,
      });
      env = "LOCAL";
    }
    if (!projectId) {
      logger({
        category: "init",
        message:
          "BROWSERBASE_PROJECT_ID is required for some Browserbase features that may not work without it.",
        level: 1,
      });
    }
  }

  if (env === "BROWSERBASE") {
    if (!apiKey) {
      throw new Error("BROWSERBASE_API_KEY is required.");
    }

    let debugUrl: string | undefined = undefined;
    let sessionUrl: string | undefined = undefined;
    let sessionId: string;
    let connectUrl: string;

    const browserbase = new Browserbase({
      apiKey,
    });

    if (browserbaseSessionID) {
      // Validate the session status
      try {
        const sessionStatus =
          await browserbase.sessions.retrieve(browserbaseSessionID);

        if (sessionStatus.status !== "RUNNING") {
          throw new Error(
            `Session ${browserbaseSessionID} is not running (status: ${sessionStatus.status})`,
          );
        }

        sessionId = browserbaseSessionID;
        const browserbaseDomain =
          BROWSERBASE_REGION_DOMAIN[sessionStatus.region] ||
          "wss://connect.browserbase.com";
        connectUrl = `${browserbaseDomain}?apiKey=${apiKey}&sessionId=${sessionId}`;

        logger({
          category: "init",
          message: "resuming existing browserbase session...",
          level: 1,
          auxiliary: {
            sessionId: {
              value: sessionId,
              type: "string",
            },
          },
        });
      } catch (error) {
        logger({
          category: "init",
          message: "failed to resume session",
          level: 1,
          auxiliary: {
            error: {
              value: error.message,
              type: "string",
            },
            trace: {
              value: error.stack,
              type: "string",
            },
          },
        });
        throw error;
      }
    } else {
      // Create new session (existing code)
      logger({
        category: "init",
        message: "creating new browserbase session...",
        level: 0,
      });

      if (!projectId) {
        throw new Error(
          "BROWSERBASE_PROJECT_ID is required for new Browserbase sessions.",
        );
      }

      const session = await browserbase.sessions.create({
        projectId,
        ...browserbaseSessionCreateParams,
      });

      sessionId = session.id;
      connectUrl = session.connectUrl;
      logger({
        category: "init",
        message: "created new browserbase session",
        level: 1,
        auxiliary: {
          sessionId: {
            value: sessionId,
            type: "string",
          },
        },
      });
    }

    const browser = await puppeteer.connect({
      browserWSEndpoint: connectUrl,
    });
    const { debuggerUrl } = await browserbase.sessions.debug(sessionId);

    debugUrl = debuggerUrl;
    sessionUrl = `https://www.browserbase.com/sessions/${sessionId}`;

    logger({
      category: "init",
      message: browserbaseSessionID
        ? "browserbase session resumed"
        : "browserbase session started",
      level: 0,
      auxiliary: {
        sessionUrl: {
          value: sessionUrl,
          type: "string",
        },
        debugUrl: {
          value: debugUrl,
          type: "string",
        },
        sessionId: {
          value: sessionId,
          type: "string",
        },
      },
    });

    const context = await browser.createBrowserContext();

    if (localBrowserLaunchOptions?.cookies) {
      await context.setCookie(...localBrowserLaunchOptions.cookies);
    }

    logger({
      category: "init",
      message: "local browser started successfully.",
    });

    return { browser, context, debugUrl, sessionUrl, sessionId, env };
  } else {
    logger({
      category: "init",
      message: "launching local browser",
      level: 0,
      auxiliary: {
        headless: {
          value: headless.toString(),
          type: "boolean",
        },
      },
    });

    if (localBrowserLaunchOptions) {
      logger({
        category: "init",
        message: "local browser launch options",
        level: 0,
        auxiliary: {
          localLaunchOptions: {
            value: JSON.stringify(localBrowserLaunchOptions),
            type: "string",
          },
        },
      });
    }

    let userDataDir = localBrowserLaunchOptions?.userDataDir;
    if (!userDataDir) {
      const tmpDirPath = path.join(os.tmpdir(), "stagehand");
      if (!fs.existsSync(tmpDirPath)) {
        fs.mkdirSync(tmpDirPath, { recursive: true });
      }

      const tmpDir = fs.mkdtempSync(path.join(tmpDirPath, "ctx_"));
      fs.mkdirSync(path.join(tmpDir, "userdir/Default"), { recursive: true });

      const defaultPreferences = {
        plugins: {
          always_open_pdf_externally: true,
        },
      };

      fs.writeFileSync(
        path.join(tmpDir, "userdir/Default/Preferences"),
        JSON.stringify(defaultPreferences),
      );
      userDataDir = path.join(tmpDir, "userdir");
    }

    let downloadsPath = localBrowserLaunchOptions?.downloadsPath;
    if (!downloadsPath) {
      downloadsPath = path.join(process.cwd(), "downloads");
      fs.mkdirSync(downloadsPath, { recursive: true });
    }

    const browser = await puppeteer.launch({
      userDataDir: userDataDir,
      headless: localBrowserLaunchOptions?.headless ?? headless,
      defaultViewport: {
        width: localBrowserLaunchOptions?.viewport?.width ?? 1024,
        height: localBrowserLaunchOptions?.viewport?.height ?? 768,
      },
      args: localBrowserLaunchOptions?.args ?? [
        "--enable-webgl",
        "--use-gl=swiftshader",
        "--enable-accelerated-2d-canvas",
        "--disable-blink-features=AutomationControlled",
        "--disable-web-security",
      ],
      devtools: localBrowserLaunchOptions?.devtools ?? false,
      env: localBrowserLaunchOptions?.env
        ? Object.entries(localBrowserLaunchOptions.env).reduce(
            (acc, [key, value]) => {
              acc[key] = String(value);
              return acc;
            },
            {} as Record<string, string>,
          )
        : undefined,
      handleSIGHUP: localBrowserLaunchOptions?.handleSIGHUP ?? true,
      handleSIGINT: localBrowserLaunchOptions?.handleSIGINT ?? true,
      handleSIGTERM: localBrowserLaunchOptions?.handleSIGTERM ?? true,
      ignoreDefaultArgs: localBrowserLaunchOptions?.ignoreDefaultArgs,
      executablePath: localBrowserLaunchOptions?.executablePath,
      channel: "chrome",
    });

    const context = await browser.createBrowserContext();

    if (localBrowserLaunchOptions?.cookies) {
      await context.setCookie(...localBrowserLaunchOptions.cookies);
    }

    logger({
      category: "init",
      message: "local browser started successfully.",
    });

    return { context, contextPath: userDataDir, env: "LOCAL" };
  }
}

const defaultLogger = async (logLine: LogLine) => {
  console.log(logLineToString(logLine));
};

export class Stagehand {
  private stagehandPage!: StagehandPage;
  private stagehandContext!: StagehandContext;
  private intEnv: "LOCAL" | "BROWSERBASE";

  public browserbaseSessionID?: string;
  public readonly domSettleTimeoutMs: number;
  public readonly debugDom: boolean;
  public readonly headless: boolean;
  public verbose: 0 | 1 | 2;
  public llmProvider: LLMProvider;
  public enableCaching: boolean;

  private apiKey: string | undefined;
  private projectId: string | undefined;
  private externalLogger?: (logLine: LogLine) => void;
  private browserbaseSessionCreateParams?: Browserbase.Sessions.SessionCreateParams;
  public variables: { [key: string]: unknown };
  private contextPath?: string;
  public llmClient: LLMClient;
  public readonly userProvidedInstructions?: string;
  private usingAPI: boolean;
  private modelName: AvailableModel;
  public apiClient: StagehandAPI | undefined;
  public readonly waitForCaptchaSolves: boolean;
  private localBrowserLaunchOptions?: LocalBrowserLaunchOptions;
  public readonly selfHeal: boolean;
  private cleanupCalled = false;
  public readonly actTimeoutMs: number;
  public readonly logInferenceToFile?: boolean;
  protected setActivePage(page: StagehandPage): void {
    this.stagehandPage = page;
  }

  public get page(): Page {
    if (!this.stagehandContext) {
      throw new Error(
        "Stagehand not initialized. Make sure to await stagehand.init() first.",
      );
    }

    return this.stagehandPage.page;
  }

  public stagehandMetrics: StagehandMetrics = {
    actPromptTokens: 0,
    actCompletionTokens: 0,
    actInferenceTimeMs: 0,
    extractPromptTokens: 0,
    extractCompletionTokens: 0,
    extractInferenceTimeMs: 0,
    observePromptTokens: 0,
    observeCompletionTokens: 0,
    observeInferenceTimeMs: 0,
    totalPromptTokens: 0,
    totalCompletionTokens: 0,
    totalInferenceTimeMs: 0,
  };

  public get metrics(): StagehandMetrics {
    return this.stagehandMetrics;
  }

  public updateMetrics(
    functionName: StagehandFunctionName,
    promptTokens: number,
    completionTokens: number,
    inferenceTimeMs: number,
  ): void {
    switch (functionName) {
      case StagehandFunctionName.ACT:
        this.stagehandMetrics.actPromptTokens += promptTokens;
        this.stagehandMetrics.actCompletionTokens += completionTokens;
        this.stagehandMetrics.actInferenceTimeMs += inferenceTimeMs;
        break;

      case StagehandFunctionName.EXTRACT:
        this.stagehandMetrics.extractPromptTokens += promptTokens;
        this.stagehandMetrics.extractCompletionTokens += completionTokens;
        this.stagehandMetrics.extractInferenceTimeMs += inferenceTimeMs;
        break;

      case StagehandFunctionName.OBSERVE:
        this.stagehandMetrics.observePromptTokens += promptTokens;
        this.stagehandMetrics.observeCompletionTokens += completionTokens;
        this.stagehandMetrics.observeInferenceTimeMs += inferenceTimeMs;
        break;
    }
    this.updateTotalMetrics(promptTokens, completionTokens, inferenceTimeMs);
  }

  private updateTotalMetrics(
    promptTokens: number,
    completionTokens: number,
    inferenceTimeMs: number,
  ): void {
    this.stagehandMetrics.totalPromptTokens += promptTokens;
    this.stagehandMetrics.totalCompletionTokens += completionTokens;
    this.stagehandMetrics.totalInferenceTimeMs += inferenceTimeMs;
  }

  constructor(
    {
      env,
      apiKey,
      projectId,
      verbose,
      debugDom,
      llmProvider,
      llmClient,
      headless,
      logger,
      browserbaseSessionCreateParams,
      domSettleTimeoutMs,
      enableCaching,
      browserbaseSessionID,
      modelName,
      modelClientOptions,
      systemPrompt,
      useAPI,
      localBrowserLaunchOptions,
      selfHeal = true,
      waitForCaptchaSolves = false,
      actTimeoutMs = 60_000,
      logInferenceToFile = false,
    }: ConstructorParams = {
      env: "BROWSERBASE",
    },
  ) {
    this.externalLogger = logger || defaultLogger;
    this.enableCaching =
      enableCaching ??
      (process.env.ENABLE_CACHING && process.env.ENABLE_CACHING === "true");
    this.llmProvider =
      llmProvider || new LLMProvider(this.logger, this.enableCaching);
    this.intEnv = env;
    this.apiKey = apiKey ?? process.env.BROWSERBASE_API_KEY;
    this.projectId = projectId ?? process.env.BROWSERBASE_PROJECT_ID;
    this.verbose = verbose ?? 0;
    this.debugDom = debugDom ?? false;
    if (llmClient) {
      this.llmClient = llmClient;
    } else {
      try {
        // try to set a default LLM client
        this.llmClient = this.llmProvider.getClient(
          modelName ?? DEFAULT_MODEL_NAME,
          modelClientOptions,
        );
      } catch {
        this.llmClient = undefined;
      }
    }

    this.domSettleTimeoutMs = domSettleTimeoutMs ?? 30_000;
    this.headless = headless ?? false;
    this.browserbaseSessionCreateParams = browserbaseSessionCreateParams;
    this.browserbaseSessionID = browserbaseSessionID;
    this.userProvidedInstructions = systemPrompt;
    this.usingAPI = useAPI ?? false;
    this.modelName = modelName ?? DEFAULT_MODEL_NAME;
    this.actTimeoutMs = actTimeoutMs;

    if (this.usingAPI && env === "LOCAL") {
      throw new Error("API mode can only be used with BROWSERBASE environment");
    } else if (this.usingAPI && !process.env.STAGEHAND_API_URL) {
      throw new Error(
        "STAGEHAND_API_URL is required when using the API. Please set it in your environment variables.",
      );
    } else if (
      this.usingAPI &&
      this.llmClient.type !== "openai" &&
      this.llmClient.type !== "anthropic"
    ) {
      throw new Error(
        "API mode requires an OpenAI or Anthropic LLM. Please provide a compatible model.",
      );
    }
    this.waitForCaptchaSolves = waitForCaptchaSolves;

    this.selfHeal = selfHeal;
    this.localBrowserLaunchOptions = localBrowserLaunchOptions;

    if (this.usingAPI) {
      this.registerSignalHandlers();
    }
    this.logInferenceToFile = logInferenceToFile;
  }

  private registerSignalHandlers() {
    const cleanup = async (signal: string) => {
      if (this.cleanupCalled) return;
      this.cleanupCalled = true;

      console.log(`[${signal}] received. Ending Browserbase session...`);
      try {
        await this.close();
      } catch (err) {
        console.error("Error ending Browserbase session:", err);
      } finally {
        // Exit explicitly once cleanup is done
        process.exit(0);
      }
    };

    process.once("SIGINT", () => void cleanup("SIGINT"));
    process.once("SIGTERM", () => void cleanup("SIGTERM"));
  }

  public get logger(): (logLine: LogLine) => void {
    return (logLine: LogLine) => {
      this.log(logLine);
    };
  }

  public get env(): "LOCAL" | "BROWSERBASE" {
    if (this.intEnv === "BROWSERBASE" && this.apiKey && this.projectId) {
      return "BROWSERBASE";
    }
    return "LOCAL";
  }

  public get context(): EnhancedContext {
    if (!this.stagehandContext) {
      throw new Error(
        "Stagehand not initialized. Make sure to await stagehand.init() first.",
      );
    }
    return this.stagehandContext.context;
  }

  async init(
    /** @deprecated Use constructor options instead */
    initOptions?: InitOptions,
  ): Promise<InitResult> {
    if (initOptions) {
      console.warn(
        "Passing parameters to init() is deprecated and will be removed in the next major version. Use constructor options instead.",
      );
    }

    if (this.usingAPI) {
      this.apiClient = new StagehandAPI({
        apiKey: this.apiKey,
        projectId: this.projectId,
        logger: this.logger,
      });

      const { sessionId } = await this.apiClient.init({
        modelName: this.modelName,
        modelApiKey:
          LLMProvider.getModelProvider(this.modelName) === "openai"
            ? process.env.OPENAI_API_KEY
            : process.env.ANTHROPIC_API_KEY,
        domSettleTimeoutMs: this.domSettleTimeoutMs,
        verbose: this.verbose,
        debugDom: this.debugDom,
        systemPrompt: this.userProvidedInstructions,
        selfHeal: this.selfHeal,
        waitForCaptchaSolves: this.waitForCaptchaSolves,
        actionTimeoutMs: this.actTimeoutMs,
        browserbaseSessionCreateParams: this.browserbaseSessionCreateParams,
      });
      this.browserbaseSessionID = sessionId;
    }

    const { context, debugUrl, sessionUrl, contextPath, sessionId, env } =
      await getBrowser(
        this.apiKey,
        this.projectId,
        this.env,
        this.headless,
        this.logger,
        this.browserbaseSessionCreateParams,
        this.browserbaseSessionID,
        this.localBrowserLaunchOptions,
      ).catch((e) => {
        console.error("Error in init:", e);
        const br: BrowserResult = {
          context: undefined,
          debugUrl: undefined,
          sessionUrl: undefined,
          sessionId: undefined,
          env: this.env,
        };
        return br;
      });
    this.intEnv = env;
    this.contextPath = contextPath;

    this.stagehandContext = await StagehandContext.init(context, this);

    const defaultPage = (await this.stagehandContext.getStagehandPages())[0];
    this.stagehandPage = defaultPage;

    if (this.headless) {
      await this.page.setViewport({ width: 1280, height: 720 });
    }

    this.browserbaseSessionID = sessionId;

    return { debugUrl, sessionUrl, sessionId };
  }

  private pending_logs_to_send_to_browserbase: LogLine[] = [];

  private is_processing_browserbase_logs: boolean = false;

  log(logObj: LogLine): void {
    logObj.level = logObj.level ?? 1;

    if (this.externalLogger) {
      this.externalLogger(logObj);
    }

    this.pending_logs_to_send_to_browserbase.push({
      ...logObj,
      id: randomUUID(),
    });
    this._run_browserbase_log_processing_cycle();
  }

  private async _run_browserbase_log_processing_cycle() {
    if (this.is_processing_browserbase_logs) {
      return;
    }
    this.is_processing_browserbase_logs = true;
    const pending_logs = [...this.pending_logs_to_send_to_browserbase];
    for (const logObj of pending_logs) {
      await this._log_to_browserbase(logObj);
    }
    this.is_processing_browserbase_logs = false;
  }

  private async _log_to_browserbase(logObj: LogLine) {
    logObj.level = logObj.level ?? 1;

    if (!this.stagehandPage) {
      return;
    }

    if (this.verbose >= logObj.level) {
      await this.page
        .evaluate((logObj) => {
          const logMessage = logLineToString(logObj);
          if (
            logObj.message.toLowerCase().includes("trace") ||
            logObj.message.toLowerCase().includes("error:")
          ) {
            console.error(logMessage);
          } else {
            console.log(logMessage);
          }
        }, logObj)
        .then(() => {
          this.pending_logs_to_send_to_browserbase =
            this.pending_logs_to_send_to_browserbase.filter(
              (log) => log.id !== logObj.id,
            );
        })
        .catch(() => {});
    }
  }

  /** @deprecated Use stagehand.page.act() instead. This will be removed in the next major release. */
  async act(options: ActOptions): Promise<ActResult> {
    return await this.stagehandPage.act(options);
  }

  /** @deprecated Use stagehand.page.extract() instead. This will be removed in the next major release. */
  async extract<T extends z.AnyZodObject>(
    options: ExtractOptions<T>,
  ): Promise<ExtractResult<T>> {
    return await this.stagehandPage.extract(options);
  }

  /** @deprecated Use stagehand.page.observe() instead. This will be removed in the next major release. */
  async observe(options?: ObserveOptions): Promise<ObserveResult[]> {
    return await this.stagehandPage.observe(options);
  }

  async close(): Promise<void> {
    if (this.apiClient) {
      const response = await this.apiClient.end();
      const body: ApiResponse<unknown> = await response.json();
      if (!body.success) {
        if (response.status == 409) {
          this.log({
            category: "close",
            message:
              "Warning: attempted to end a session that is not currently active",
            level: 0,
          });
        } else {
          throw new Error((body as ErrorResponse).message);
        }
      }
      return;
    } else {
      await this.context.close();
    }

    if (this.contextPath) {
      try {
        fs.rmSync(this.contextPath, { recursive: true, force: true });
      } catch (e) {
        console.error("Error deleting context directory:", e);
      }
    }
  }

  /**
   * Create an agent instance that can be executed with different instructions
   * @returns An agent instance with execute() method
   */
  agent(options: AgentConfig): {
    execute: (
      instructionOrOptions: string | AgentExecuteOptions,
    ) => Promise<AgentResult>;
  } {
    const agentHandler = new StagehandAgentHandler(
      this.stagehandPage,
      this.logger,
      {
        modelName: options.model,
        clientOptions: options.options,
        userProvidedInstructions: options.instructions,
        agentType: options.provider,
      },
    );

    this.log({
      category: "agent",
      message: "Creating agent instance",
      level: 1,
    });

    return {
      execute: async (instructionOrOptions: string | AgentExecuteOptions) => {
        const executeOptions: AgentExecuteOptions =
          typeof instructionOrOptions === "string"
            ? { instruction: instructionOrOptions }
            : instructionOrOptions;

        if (!executeOptions.instruction) {
          throw new Error("Instruction is required for agent execution");
        }

        return await agentHandler.execute(executeOptions);
      },
    };
  }
}

export * from "../types/browser";
export * from "../types/log";
export * from "../types/model";
export * from "../types/page";
export * from "../types/playwright";
export * from "../types/stagehand";
export * from "../types/agent";
export * from "./llm/LLMClient";

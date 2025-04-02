import { EvalFunction } from "@/types/evals";
import { initStagehand } from "@/evals/initStagehand";
import { z } from "zod";
import { openai } from "@ai-sdk/openai/dist";
import { AISdkClient } from "@/examples/external_clients/aisdk";

export const hn_aisdk: EvalFunction = async ({ logger }) => {
  const { stagehand, initResponse } = await initStagehand({
    logger,
    llmClient: new AISdkClient({
      model: openai("gpt-4o-mini"),
    }),
  });

  const { debugUrl, sessionUrl } = initResponse;

  await stagehand.page.goto("https://news.ycombinator.com");

  let { story } = await stagehand.page.extract({
    instruction: "extract the title of the top story on the page",
    schema: z.object({
      story: z.string().describe("the title of the top story on the page"),
    }),
  });
  // remove the (url) part of the story title
  story = story.split(" (")[0];

  const expectedStoryElement = await stagehand.page.$(
    "xpath=/html/body/center/table/tbody/tr[3]/td/table/tbody/tr[1]/td[3]/span/a",
  );
  // remove the (url) part of the story title
  const expectedStory = (await expectedStoryElement?.textContent())?.split(
    " (",
  )?.[0];

  if (!expectedStory) {
    logger.error({
      message: "Could not find expected story element",
      level: 0,
    });
    return {
      _success: false,
      error: "Could not find expected story element",
      debugUrl,
      sessionUrl,
      logs: logger.getLogs(),
    };
  }

  if (story !== expectedStory) {
    logger.error({
      message: "Extracted story does not match expected story",
      level: 0,
      auxiliary: {
        expected: {
          value: expectedStory,
          type: "string",
        },
        actual: {
          value: story,
          type: "string",
        },
      },
    });
    return {
      _success: false,
      error: "Extracted story does not match expected story",
      expectedStory,
      actualStory: story,
      debugUrl,
      sessionUrl,
      logs: logger.getLogs(),
    };
  }

  await stagehand.page.act("Click on the 'new' tab");

  if (stagehand.page.url() !== "https://news.ycombinator.com/newest") {
    logger.error({
      message: "Page did not navigate to the 'new' tab",
      level: 0,
      auxiliary: {
        expected: {
          value: "https://news.ycombinator.com/newest",
          type: "string",
        },
        actual: {
          value: stagehand.page.url(),
          type: "string",
        },
      },
    });
    return {
      _success: false,
      error: "Page did not navigate to the 'new' tab",
      expectedUrl: "https://news.ycombinator.com/newest",
      actualUrl: stagehand.page.url(),
      debugUrl,
      sessionUrl,
      logs: logger.getLogs(),
    };
  }

  await stagehand.close();

  return {
    _success: true,
    expectedStory,
    actualStory: story,
    debugUrl,
    sessionUrl,
    logs: logger.getLogs(),
  };
};

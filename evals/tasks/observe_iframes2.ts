import { initStagehand } from "@/evals/initStagehand";
import { EvalFunction } from "@/types/evals";

export const observe_iframes2: EvalFunction = async ({ modelName, logger }) => {
  const { stagehand, debugUrl, sessionUrl } = await initStagehand({
    modelName,
    logger,
  });

  await stagehand.page.goto(
    "https://iframetester.com/?url=https://shopify.com",
  );
  await new Promise((resolve) => setTimeout(resolve, 5000));

  const observations = await stagehand.page.observe({
    instruction: "find the main header of the page",
  });

  if (observations.length === 0) {
    await stagehand.close();
    return {
      _success: false,
      observations,
      debugUrl,
      sessionUrl,
      logs: logger.getLogs(),
    };
  }

  const possibleLocators = [`#iframe-window`, `body > header > h1`];

  const possibleHandles = [];
  for (const locatorStr of possibleLocators) {
    const locator = stagehand.page.locator(locatorStr);
    const handle = await locator.elementHandle();
    if (handle) {
      possibleHandles.push({ locatorStr, handle });
    }
  }

  let foundMatch = false;
  let matchedLocator: string | null = null;

  for (const observation of observations) {
    try {
      const observationLocator = stagehand.page
        .locator(observation.selector)
        .first();
      const observationHandle = await observationLocator.elementHandle();
      if (!observationHandle) {
        continue;
      }

      for (const { locatorStr, handle: candidateHandle } of possibleHandles) {
        const isSameNode = await observationHandle.evaluate(
          (node, otherNode) => node === otherNode,
          candidateHandle,
        );
        if (isSameNode) {
          foundMatch = true;
          matchedLocator = locatorStr;
          break;
        }
      }

      if (foundMatch) {
        break;
      }
    } catch (error) {
      console.warn(
        `Failed to check observation with selector ${observation.selector}:`,
        error.message,
      );
      continue;
    }
  }

  await stagehand.close();

  return {
    _success: foundMatch,
    matchedLocator,
    observations,
    debugUrl,
    sessionUrl,
    logs: logger.getLogs(),
  };
};

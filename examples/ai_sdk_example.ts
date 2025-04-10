import { openai } from "@ai-sdk/openai";
import { Stagehand } from "@/dist";
import { AISdkClient } from "./external_clients/aisdk";
import StagehandConfig from "@/stagehand.config";
import { z } from "zod";

async function example() {
  const stagehand = new Stagehand({
    ...StagehandConfig,
    llmClient: new AISdkClient({
      model: openai("gpt-4o"),
    }),
  });

  await stagehand.init();
  await stagehand.page.goto("https://www.avto.net/");

  //await stagehand.page.waitForLoadState("networkidle");

  await stagehand.page.act("Click on a button with text 'Dovoli piškotke'");

  await stagehand.page.act("Select the search filters: Brand - Mercedes-Benz");

  await stagehand.page.act("Click on a button with text 'Iskanje vozil'");

  const data = await stagehand.page.extract({
    instruction:
      "Extract all the listings in the first page of the search results",
    schema: z.object({
      listings: z
        .array(
          z.object({
            title: z.string().describe("The title of the listing"),
            price: z.string().describe("The price of the listing"),
            image: z.string().describe("The image of the listing"),
            link: z.string().describe("The link of the listing"),
          }),
        )
        .describe("The listings in the first page of the search results"),
    }),
  });

  console.log(data);

  //await stagehand.close();
}

(async () => {
  await example();
})();

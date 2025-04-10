import { Stagehand } from "@/dist";
//import { z } from "zod";
import StagehandConfig from "@/stagehand.config";

async function runGoogleExample() {
  console.log("Initializing Stagehand with Google Client");
  // Initialize Stagehand with the Google configuration
  const stagehand = new Stagehand({
    ...StagehandConfig,
    modelName: "gemini-2.0-flash",
    modelClientOptions: {
      apiKey: process.env.GOOGLE_API_KEY,
      userProvidedInstructions:
        "You are a helpful assistant navigating https://www.avto.net/.",
    },
  });

  await stagehand.init();
  await stagehand.page.goto("https://www.avto.net/");

  // Allow some time for the page to load
  await stagehand.page.waitForLoadState("networkidle");

  // Use the extract method to get car brand options
  await stagehand.page.act("Click on a button with text 'Dovoli piškotke'");

  // await stagehand.page.act(
  //   "Select the search filters: Brand - Mercedes-Benz and press on Search button",
  // );

  // const data = await stagehand.page.extract({
  //   instruction:
  //     "Extract all the listings in the first page of the search results",
  //   schema: z.object({
  //     listings: z
  //       .array(
  //         z.object({
  //           title: z.string().describe("The title of the listing"),
  //           price: z.string().describe("The price of the listing"),
  //           image: z.string().describe("The image of the listing"),
  //           link: z.string().describe("The link of the listing"),
  //         }),
  //       )
  //       .describe("The listings in the first page of the search results"),
  //   }),
  // });

  // console.log(data);

  //await stagehand.close();
}

(async () => {
  try {
    await runGoogleExample();
  } catch (error) {
    console.error("An error occurred during the example run:", error);
    process.exit(1);
  }
})();

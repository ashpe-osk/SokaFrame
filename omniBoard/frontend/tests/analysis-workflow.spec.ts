import { resolve } from "node:path"
import { expect, test } from "@playwright/test"

test("create project and queue an uploaded source video", async ({ page }) => {
  await page.goto("/")
  await expect(
    page.getByText("Welcome back, nice to see you again!!!"),
  ).toBeVisible()
  await page.getByRole("link", { name: "Open Projects" }).click()

  await expect(
    page.getByRole("heading", { name: "Projects", exact: true }),
  ).toBeVisible()
  await page.getByLabel("Project name").fill(`Playwright ${Date.now()}`)
  await page.getByRole("button", { name: "Create project" }).click()

  await expect(page).toHaveURL(/\/projects\/[0-9a-f-]+$/)
  await expect(
    page.getByText("Upload a source video", { exact: true }),
  ).toBeVisible()

  await page
    .getByLabel("Video file")
    .setInputFiles(
      resolve(process.cwd(), "..", "..", "data", "vid_for_data_15s.mp4"),
    )
  await page.getByRole("button", { name: "Upload and analyze" }).click()

  await expect(page.getByText("vid_for_data_15s.mp4")).toBeVisible({
    timeout: 30_000,
  })
  await expect(
    page.getByText(
      /^(queued|running|cancel_requested|succeeded|failed|cancelled|interrupted)$/,
    ),
  ).toBeVisible()
})

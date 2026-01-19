import { test, expect, Page } from '@playwright/test'
import path from 'path'

const FCS_FILE = path.resolve(__dirname, '../../../example_fcs_data/2026-01-11_2 recipes for T_EBFP2.fcs')

async function waitForAppReady(page: Page): Promise<void> {
  await page.waitForLoadState('domcontentloaded')
  await page.waitForSelector('text=FILES', { timeout: 60000 })
  await page.waitForTimeout(1000)
}

async function uploadFile(page: Page, filePath: string): Promise<void> {
  const fileInput = page.locator('input[type="file"][accept=".fcs"]')
  await expect(fileInput).toBeAttached({ timeout: 10000 })
  await fileInput.setInputFiles(filePath)
}

async function waitForFileLoaded(page: Page, timeout = 120000): Promise<void> {
  await page.waitForFunction(
    () => {
      const parsing = document.querySelector('*')?.textContent?.includes('Parsing')
      if (parsing) return false

      const items = document.querySelectorAll('[class*="tabular-nums"]')
      for (const item of items) {
        const text = item.textContent || ''
        const num = parseInt(text.replace(/,/g, ''), 10)
        if (!isNaN(num) && num > 100000) {
          return true
        }
      }
      return false
    },
    { timeout }
  )
  await page.waitForTimeout(2000)
}

async function canvasHasContent(page: Page): Promise<boolean> {
  const canvases = page.locator('canvas')
  const count = await canvases.count()
  if (count === 0) return false

  for (let i = 0; i < count; i++) {
    const canvas = canvases.nth(i)
    const isVisible = await canvas.isVisible()
    if (!isVisible) continue

    const hasContent = await canvas.evaluate((el: HTMLCanvasElement) => {
      const gl = el.getContext('webgl') || el.getContext('webgl2')
      if (!gl) {
        const ctx = el.getContext('2d')
        if (!ctx) return false
        const imageData = ctx.getImageData(0, 0, el.width, el.height)
        let nonWhite = 0
        for (let j = 0; j < imageData.data.length; j += 4) {
          if (imageData.data[j] < 200 || imageData.data[j+1] < 200 || imageData.data[j+2] < 200) {
            nonWhite++
          }
        }
        return nonWhite > 100
      }
      const width = gl.drawingBufferWidth
      const height = gl.drawingBufferHeight
      if (width === 0 || height === 0) return false
      const pixels = new Uint8Array(width * height * 4)
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
      let nonZeroCount = 0
      for (let j = 0; j < pixels.length; j += 4) {
        if (pixels[j] > 10 || pixels[j+1] > 10 || pixels[j+2] > 10) {
          nonZeroCount++
        }
      }
      return nonZeroCount > 100
    })
    if (hasContent) return true
  }
  return false
}

async function selectChannels(page: Page): Promise<void> {
  const xTrigger = page.locator('[role="combobox"]').first()
  await xTrigger.click()
  await page.waitForTimeout(300)

  const xOptions = page.locator('[role="option"]')
  const xCount = await xOptions.count()
  if (xCount > 0) {
    for (let i = 0; i < xCount; i++) {
      const text = await xOptions.nth(i).textContent()
      if (text && !text.includes('no channels')) {
        await xOptions.nth(i).click()
        break
      }
    }
  }
  await page.waitForTimeout(500)

  const yTrigger = page.locator('[role="combobox"]').nth(1)
  await yTrigger.click()
  await page.waitForTimeout(300)

  const yOptions = page.locator('[role="option"]')
  const yCount = await yOptions.count()
  if (yCount > 1) {
    for (let i = 1; i < yCount; i++) {
      const text = await yOptions.nth(i).textContent()
      if (text && !text.includes('no channels')) {
        await yOptions.nth(i).click()
        break
      }
    }
  }
  await page.waitForTimeout(500)
}

async function toggleFileVisibility(page: Page): Promise<void> {
  // The FileList visibility toggle is in the FILES section on the left sidebar
  // It's a button containing an Eye or EyeOff icon (lucide-react icons)
  // The button is inside a CompactListItem which shows the file name

  // First, find the FILES section
  const filesSection = page.locator('text=FILES').locator('..')

  // Find the visibility toggle button - it's a button with an SVG that has lucide-eye or lucide-eye-off class
  // The button is next to the file name in the list item
  const eyeButton = page.locator('button:has(svg.lucide-eye), button:has(svg.lucide-eye-off)').first()

  if (await eyeButton.isVisible({ timeout: 5000 }).catch(() => false)) {
    await eyeButton.click()
    return
  }

  // Fallback: try finding by structure - the button with Eye icon near the file count badge
  const fileListItem = page.locator('[class*="tabular-nums"]').first().locator('..').locator('button').first()
  if (await fileListItem.isVisible({ timeout: 1000 }).catch(() => false)) {
    await fileListItem.click()
    return
  }

  throw new Error('Could not find FileList visibility toggle button')
}

test.describe('DensityPlot Visibility Toggle', () => {
  test.describe.configure({ mode: 'serial' })

  test.beforeEach(async ({ page }) => {
    page.on('console', msg => {
      if (msg.type() === 'error' || msg.text().includes('WebSocket') || msg.text().includes('Declair') || msg.text().includes('DensityPlot')) {
        console.log(`BROWSER ${msg.type()}: ${msg.text()}`)
      }
    })
    await page.goto('/')
    await waitForAppReady(page)

    await page.waitForFunction(
      () => {
        const items = document.querySelectorAll('[class*="tabular-nums"]')
        for (const item of items) {
          const text = item.textContent || ''
          const num = parseInt(text.replace(/,/g, ''), 10)
          if (!isNaN(num) && num > 1000) {
            return false
          }
        }
        return true
      },
      { timeout: 5000 }
    ).catch(() => {})
  })

  test('points reappear after hiding and showing file', async ({ page }) => {
    test.slow()

    await uploadFile(page, FCS_FILE)
    console.log('File upload initiated')

    await waitForFileLoaded(page)
    console.log('File loaded')
    await page.waitForTimeout(2000)

    await page.screenshot({ path: 'test-results/01-file-loaded.png' })

    await selectChannels(page)
    console.log('Channels selected')
    await page.waitForTimeout(3000)

    await page.screenshot({ path: 'test-results/02-channels-selected.png' })

    const canvas = page.locator('canvas').first()
    await expect(canvas).toBeVisible()

    const initialHasContent = await canvasHasContent(page)
    console.log('Initial canvas has content:', initialHasContent)
    await page.screenshot({ path: 'test-results/03-initial-state.png' })
    expect(initialHasContent).toBe(true)

    await toggleFileVisibility(page)
    console.log('Toggled visibility off')
    await page.waitForTimeout(1000)
    await page.screenshot({ path: 'test-results/04-after-hide.png' })

    const afterHideHasContent = await canvasHasContent(page)
    console.log('After hide canvas has content:', afterHideHasContent)
    expect(afterHideHasContent).toBe(false)

    await toggleFileVisibility(page)
    console.log('Toggled visibility on')
    // Wait for canvas to have content (poll until computation completes)
    await page.waitForFunction(
      () => {
        const canvases = document.querySelectorAll('canvas')
        for (const canvas of canvases) {
          const gl = (canvas as HTMLCanvasElement).getContext('webgl') || (canvas as HTMLCanvasElement).getContext('webgl2')
          if (gl) {
            const width = gl.drawingBufferWidth
            const height = gl.drawingBufferHeight
            if (width > 0 && height > 0) {
              const pixels = new Uint8Array(width * height * 4)
              gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
              let nonZero = 0
              for (let j = 0; j < pixels.length; j += 4) {
                if (pixels[j] > 10 || pixels[j+1] > 10 || pixels[j+2] > 10) nonZero++
              }
              if (nonZero > 100) return true
            }
          }
        }
        return false
      },
      { timeout: 10000 }
    )
    await page.screenshot({ path: 'test-results/05-after-show.png' })

    const afterShowHasContent = await canvasHasContent(page)
    console.log('After show canvas has content:', afterShowHasContent)
    expect(afterShowHasContent).toBe(true)
  })

  test.skip('multiple hide/show cycles work correctly', async ({ page }) => {
    test.slow()

    await uploadFile(page, FCS_FILE)
    await waitForFileLoaded(page)

    await selectChannels(page)
    await page.waitForTimeout(5000)

    await page.waitForFunction(
      async () => {
        const canvases = document.querySelectorAll('canvas')
        for (const canvas of canvases) {
          const gl = canvas.getContext('webgl') || canvas.getContext('webgl2')
          if (gl) {
            const width = gl.drawingBufferWidth
            const height = gl.drawingBufferHeight
            if (width > 0 && height > 0) {
              const pixels = new Uint8Array(width * height * 4)
              gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
              let nonZero = 0
              for (let j = 0; j < pixels.length; j += 4) {
                if (pixels[j] > 10 || pixels[j+1] > 10 || pixels[j+2] > 10) nonZero++
              }
              if (nonZero > 100) return true
            }
          }
        }
        return false
      },
      { timeout: 30000 }
    )

    for (let cycle = 0; cycle < 3; cycle++) {
      console.log(`Cycle ${cycle + 1}`)
      const beforeHide = await canvasHasContent(page)
      expect(beforeHide).toBe(true)

      await toggleFileVisibility(page)
      await page.waitForTimeout(500)

      const afterHide = await canvasHasContent(page)
      expect(afterHide).toBe(false)

      await toggleFileVisibility(page)
      await page.waitForTimeout(500)

      const afterShow = await canvasHasContent(page)
      expect(afterShow).toBe(true)
    }
  })
})

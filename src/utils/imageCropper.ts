export interface CropRegion {
  x: number
  y: number
  width: number
  height: number
}

export interface CroppedImage {
  data: string
  region: CropRegion
  gridIndex: number
}

export class ImageCropper {
  static cropImageRegion(
    imageUrl: string,
    region: CropRegion,
    onComplete: (croppedData: string) => void,
    onError: (error: string) => void
  ): void {
    const img = new Image()
    img.crossOrigin = 'anonymous'

    img.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = region.width
        canvas.height = region.height

        const ctx = canvas.getContext('2d')
        if (!ctx) {
          onError('Failed to get canvas context')
          return
        }

        ctx.drawImage(
          img,
          region.x,
          region.y,
          region.width,
          region.height,
          0,
          0,
          region.width,
          region.height
        )

        const croppedData = canvas.toDataURL('image/png')
        onComplete(croppedData)
      } catch (error) {
        onError(`Crop failed: ${error}`)
      }
    }

    img.onerror = () => {
      onError('Failed to load image')
    }

    img.src = imageUrl
  }

  static getGridSegmentRegion(imageWidth: number, imageHeight: number, gridIndex: number): CropRegion {
    const gridSize = 3
    const segmentWidth = imageWidth / gridSize
    const segmentHeight = imageHeight / gridSize

    const row = Math.floor(gridIndex / gridSize)
    const col = gridIndex % gridSize

    const x = col * segmentWidth
    const y = row * segmentHeight
    const width = (col === gridSize - 1) ? imageWidth - x : segmentWidth
    const height = (row === gridSize - 1) ? imageHeight - y : segmentHeight

    return { x, y, width, height }
  }

  static getAllGridSegments(imageWidth: number, imageHeight: number): CropRegion[] {
    return Array.from({ length: 9 }, (_, i) =>
      this.getGridSegmentRegion(imageWidth, imageHeight, i)
    )
  }

  static downloadCroppedImage(croppedDataUrl: string, filename: string): void {
    const link = document.createElement('a')
    link.href = croppedDataUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  static async cropAndStoreDefect(
    testImageUrl: string,
    gridIndex: number,
    imageWidth: number,
    imageHeight: number
  ): Promise<CroppedImage> {
    return new Promise((resolve, reject) => {
      const region = this.getGridSegmentRegion(imageWidth, imageHeight, gridIndex)

      this.cropImageRegion(
        testImageUrl,
        region,
        (data) => {
          resolve({
            data,
            region,
            gridIndex,
          })
        },
        reject
      )
    })
  }
}

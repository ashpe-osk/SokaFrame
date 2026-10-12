export function Footer() {
  const currentYear = new Date().getFullYear()

  return (
    <footer className="border-t py-4 px-6">
      <div className="flex flex-col items-center justify-between gap-2 sm:flex-row">
        <p className="text-muted-foreground text-sm">
          <span className="font-medium text-foreground">SokaFrame</span>
          <span className="px-2">·</span>
          Football video analysis
        </p>
        <span className="text-xs text-muted-foreground">© {currentYear}</span>
      </div>
    </footer>
  )
}

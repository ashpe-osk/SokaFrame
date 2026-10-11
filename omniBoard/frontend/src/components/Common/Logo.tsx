import { Link } from "@tanstack/react-router"

import { cn } from "@/lib/utils"

interface LogoProps {
  variant?: "full" | "icon" | "responsive"
  className?: string
  asLink?: boolean
}

function Wordmark({ className }: { className?: string }) {
  return (
    <>
      <img
        alt="SokaFrame"
        className={cn("h-9 w-auto dark:hidden", className)}
        src="/assets/images/sokaframe-logo-light.png"
      />
      <img
        alt="SokaFrame"
        className={cn("hidden h-9 w-auto dark:block", className)}
        src="/assets/images/sokaframe-logo-dark.png"
      />
    </>
  )
}

function AppIcon({
  className,
  variant = "sidebar",
}: {
  className?: string
  variant?: "sidebar" | "standalone"
}) {
  return (
    <img
      alt="SokaFrame"
      className={cn("size-8 shrink-0", className)}
      src={
        variant === "standalone"
          ? "/assets/images/sokaframe-icon.png"
          : "/assets/images/sokaframe-icon-sidebar.png"
      }
    />
  )
}

export function Logo({
  variant = "full",
  className,
  asLink = true,
}: LogoProps) {
  const content = (
    <span
      className={cn(
        "inline-flex min-w-0 items-center",
        variant === "responsive" &&
          "group-data-[collapsible=icon]:justify-center",
        className,
      )}
    >
      {variant === "icon" ? (
        <AppIcon variant="standalone" />
      ) : variant === "responsive" ? (
        <>
          <Wordmark className="h-10 max-w-full group-data-[collapsible=icon]:hidden" />
          <AppIcon className="hidden group-data-[collapsible=icon]:block group-data-[collapsible=icon]:size-9" />
        </>
      ) : (
        <Wordmark
          className={variant === "full" ? "h-16 max-w-full" : "h-9 max-w-full"}
        />
      )}
    </span>
  )

  if (!asLink) {
    return content
  }

  return <Link to="/">{content}</Link>
}

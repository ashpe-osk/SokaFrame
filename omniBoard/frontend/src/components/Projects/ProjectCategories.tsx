import { Link } from "@tanstack/react-router"
import { ClipboardList, ScanLine, Video } from "lucide-react"

type ProjectCategory = "computer-vision" | "tactical-board" | "telestrator"

export function ProjectCategories({ active }: { active: ProjectCategory }) {
  const linkClass = (category: ProjectCategory) =>
    `inline-flex items-center gap-2 rounded-lg border px-4 py-2.5 text-sm font-medium transition-colors ${
      active === category
        ? "border-primary/40 bg-primary/10 text-primary"
        : "border-border/80 bg-card text-muted-foreground hover:bg-accent hover:text-foreground"
    }`

  return (
    <nav aria-label="Project categories" className="flex flex-wrap gap-2">
      <Link
        aria-current={active === "computer-vision" ? "page" : undefined}
        className={linkClass("computer-vision")}
        to="/projects"
      >
        <ScanLine className="size-4" />
        Computer Vision
      </Link>
      <Link
        aria-current={active === "tactical-board" ? "page" : undefined}
        className={linkClass("tactical-board")}
        search={{ projectId: undefined, projectType: "board" }}
        to="/projects/tactical-board"
      >
        <ClipboardList className="size-4" />
        Tactical Board
      </Link>
      <Link
        aria-current={active === "telestrator" ? "page" : undefined}
        className={linkClass("telestrator")}
        search={{ projectId: undefined }}
        to="/projects/telestrator"
      >
        <Video className="size-4" />
        Telestrator
      </Link>
    </nav>
  )
}

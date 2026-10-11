import { createFileRoute, Link } from "@tanstack/react-router"
import { FolderKanban } from "lucide-react"

import { Button } from "@/components/ui/button"
import useAuth from "@/hooks/useAuth"

export const Route = createFileRoute("/_layout/")({
  component: Dashboard,
  head: () => ({
    meta: [
      {
        title: "Dashboard | SokaFrame",
      },
    ],
  }),
})

function Dashboard() {
  const { user: currentUser } = useAuth()

  return (
    <div>
      <div>
        <h1 className="text-2xl truncate max-w-sm">
          Hi, {currentUser?.full_name || currentUser?.email} 👋
        </h1>
        <p className="text-muted-foreground">
          Welcome back to SokaFrame. Your match workspace is ready.
        </p>
        <Button asChild className="mt-6">
          <Link to="/projects">
            <FolderKanban />
            Open Projects
          </Link>
        </Button>
      </div>
    </div>
  )
}

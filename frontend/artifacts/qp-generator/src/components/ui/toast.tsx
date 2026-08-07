import * as React from "react"
import { cn } from "@/lib/utils"

function ToastProvider({
  children,
}: {
  children: React.ReactNode
}) {
  return <>{children}</>
}

export { ToastProvider }
// Skipping full toast impl for brevity since we're mostly using sonner usually, or I'll just use a simple mock if needed.
// Wait, the template has `<Toaster />` from `@/components/ui/toaster`. I'll just create a basic one or use sonner if it's there.
// The package.json has `sonner`, so I'll write a sonner toaster.

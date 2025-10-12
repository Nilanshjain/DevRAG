import * as React from "react"
import { cn } from "../../lib/utils"

const Textarea = React.forwardRef(({ className, ...props }, ref) => {
  return (
    <textarea
      className={cn(
        "flex min-h-[80px] w-full rounded-md border-2 border-metal-600 bg-steel-900 px-3 py-2 text-sm text-metal-100 ring-offset-background placeholder:text-metal-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rust-600 focus-visible:ring-offset-2 focus-visible:border-rust-500 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-200 shadow-inner hover:border-metal-500 resize-none",
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
Textarea.displayName = "Textarea"

export { Textarea }

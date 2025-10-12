import * as React from "react"
import { cn } from "../../lib/utils"

const Input = React.forwardRef(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(
        "flex h-10 w-full rounded-md border-2 border-metal-600 bg-steel-900 px-3 py-2 text-sm text-metal-100 ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-metal-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rust-600 focus-visible:ring-offset-2 focus-visible:border-rust-500 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-200 shadow-inner hover:border-metal-500",
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
Input.displayName = "Input"

export { Input }

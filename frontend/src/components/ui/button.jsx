import * as React from "react"
import { cva } from "class-variance-authority"
import { cn } from "../../lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 relative overflow-hidden group",
  {
    variants: {
      variant: {
        default:
          "bg-metal-gradient text-white shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] before:absolute before:inset-0 before:bg-steel-shine before:opacity-0 hover:before:opacity-100 before:animate-shimmer",
        destructive:
          "bg-rust-gradient text-white shadow hover:opacity-90",
        outline:
          "border-2 border-metal-500 bg-transparent text-metal-100 hover:bg-metal-800 hover:border-rust-600 shadow-sm hover:shadow-lg transition-all duration-300",
        secondary:
          "bg-steel-800 text-metal-100 shadow hover:bg-steel-700 hover:shadow-lg",
        ghost:
          "hover:bg-metal-800 hover:text-metal-100 transition-colors duration-200",
        link:
          "text-rust-500 underline-offset-4 hover:underline hover:text-rust-400",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-9 rounded-md px-3",
        lg: "h-11 rounded-md px-8 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

const Button = React.forwardRef(({ className, variant, size, asChild = false, ...props }, ref) => {
  return (
    <button
      className={cn(buttonVariants({ variant, size, className }))}
      ref={ref}
      {...props}
    />
  )
})
Button.displayName = "Button"

export { Button, buttonVariants }

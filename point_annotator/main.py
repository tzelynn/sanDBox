import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json

class ImagePolygonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Polygon Marker")
        
        # Set up variables
        self.image_path = None
        self.original_image = None
        self.displayed_image = None
        self.image_tk = None
        self.zoom_factor = 1.0
        self.points = []
        self.point_markers = []
        self.selected_point = None
        self.lines = []
        
        # Create main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create top frame for buttons
        self.top_frame = tk.Frame(self.main_frame)
        self.top_frame.pack(fill=tk.X)
        
        # Create buttons
        self.open_button = tk.Button(self.top_frame, text="Open Image", command=self.open_image)
        self.open_button.pack(side=tk.LEFT, padx=5)
        
        self.zoom_in_button = tk.Button(self.top_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)
        
        self.zoom_out_button = tk.Button(self.top_frame, text="Zoom Out", command=self.zoom_out)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = tk.Button(self.top_frame, text="Reset Points", command=self.reset_points)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.save_button = tk.Button(self.top_frame, text="Save Points", command=self.save_points)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Create canvas with scrollbars for panning when zoomed
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.h_scrollbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.v_scrollbar = tk.Scrollbar(self.canvas_frame)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            bg="lightgray",
            xscrollcommand=self.h_scrollbar.set,
            yscrollcommand=self.v_scrollbar.set
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.h_scrollbar.config(command=self.canvas.xview)
        self.v_scrollbar.config(command=self.canvas.yview)
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.root.bind("<KeyPress>", self.on_key_press)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows and MacOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Open an image to begin.")
        self.status_bar = tk.Label(self.main_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Display instructions
        self.display_instructions()
    
    def display_instructions(self):
        """Display instructions in a message box"""
        instructions = (
            "Instructions:\n\n"
            "1. Click 'Open Image' to load an image\n"
            "2. Use 'Zoom In' and 'Zoom Out' or the mouse wheel to adjust the view\n"
            "3. Click on the image to place points (up to 4)\n"
            "4. Click on a point to select it\n"
            "5. Use arrow keys to fine-tune the selected point's position\n"
            "6. Once you have 4 points, click 'Save Points' to save the coordinates\n"
            "7. Click 'Reset Points' to start over"
        )
        messagebox.showinfo("Instructions", instructions)
    
    def open_image(self):
        """Open an image file and display it on the canvas"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.original_image = Image.open(file_path)
                self.reset_points()
                self.zoom_factor = 1.0
                self.display_image()
                self.status_var.set(f"Image loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image: {str(e)}")
    
    def display_image(self):
        """Display the image on the canvas with current zoom level"""
        if self.original_image:
            # Resize image according to zoom factor
            new_width = int(self.original_image.width * self.zoom_factor)
            new_height = int(self.original_image.height * self.zoom_factor)
            
            try:
                self.displayed_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            except AttributeError:
                # Fallback for older versions of PIL
                self.displayed_image = self.original_image.resize((new_width, new_height), Image.ANTIALIAS)
                
            self.image_tk = ImageTk.PhotoImage(self.displayed_image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            self.point_markers = []
            self.lines = []
            
            # Configure canvas scrolling region
            self.canvas.config(scrollregion=(0, 0, new_width, new_height))
            
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk, tags="image")
            
            # Redraw points and lines
            self.redraw_points()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel events for zooming"""
        if not self.original_image:
            return
        
        # Get the current position of the mouse
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Calculate the position in the original image coordinates
        orig_x = x / self.zoom_factor
        orig_y = y / self.zoom_factor
        
        # Determine zoom direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            # Zoom in
            self.zoom_factor *= 1.1
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            # Zoom out
            if self.zoom_factor > 0.1:  # Prevent excessive zooming out
                self.zoom_factor /= 1.1
        
        # Redisplay image
        self.display_image()
        
        # Calculate the new position of the point in the zoomed coordinates
        new_x = orig_x * self.zoom_factor
        new_y = orig_y * self.zoom_factor
        
        # Scroll to center on the mouse position
        self.canvas.xview_moveto((new_x - event.x) / (self.original_image.width * self.zoom_factor))
        self.canvas.yview_moveto((new_y - event.y) / (self.original_image.height * self.zoom_factor))
        
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
    
    def zoom_in(self, event=None):
        """Increase zoom factor and redisplay image"""
        if self.original_image:
            # Get the current center of the view
            x_center = self.canvas.canvasx(self.canvas.winfo_width() / 2)
            y_center = self.canvas.canvasy(self.canvas.winfo_height() / 2)
            
            # Calculate the position in the original image coordinates
            orig_x = x_center / self.zoom_factor
            orig_y = y_center / self.zoom_factor
            
            # Increase zoom factor
            self.zoom_factor *= 1.2
            
            # Redisplay image
            self.display_image()
            
            # Calculate the new center in the zoomed coordinates
            new_x = orig_x * self.zoom_factor
            new_y = orig_y * self.zoom_factor
            
            # Scroll to the new center
            self.canvas.xview_moveto((new_x - self.canvas.winfo_width() / 2) / 
                                     (self.original_image.width * self.zoom_factor))
            self.canvas.yview_moveto((new_y - self.canvas.winfo_height() / 2) / 
                                     (self.original_image.height * self.zoom_factor))
            
            self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
    
    def zoom_out(self, event=None):
        """Decrease zoom factor and redisplay image"""
        if self.original_image:
            if self.zoom_factor > 0.1:  # Prevent excessive zooming out
                # Get the current center of the view
                x_center = self.canvas.canvasx(self.canvas.winfo_width() / 2)
                y_center = self.canvas.canvasy(self.canvas.winfo_height() / 2)
                
                # Calculate the position in the original image coordinates
                orig_x = x_center / self.zoom_factor
                orig_y = y_center / self.zoom_factor
                
                # Decrease zoom factor
                self.zoom_factor /= 1.2
                
                # Redisplay image
                self.display_image()
                
                # Calculate the new center in the zoomed coordinates
                new_x = orig_x * self.zoom_factor
                new_y = orig_y * self.zoom_factor
                
                # Scroll to the new center
                self.canvas.xview_moveto((new_x - self.canvas.winfo_width() / 2) / 
                                         (self.original_image.width * self.zoom_factor))
                self.canvas.yview_moveto((new_y - self.canvas.winfo_height() / 2) / 
                                         (self.original_image.height * self.zoom_factor))
                
                self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
    
    def on_canvas_click(self, event):
        """Handle canvas click to add or select points"""
        if not self.original_image:
            return
        
        # Convert canvas coordinates to original image coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Check if clicking on an existing point (for selection)
        for i, (point_x, point_y) in enumerate(self.points):
            # Scale point coordinates according to zoom
            scaled_x = point_x * self.zoom_factor
            scaled_y = point_y * self.zoom_factor
            
            # If click is within 10 pixels of a point, select it
            if abs(scaled_x - canvas_x) < 10 and abs(scaled_y - canvas_y) < 10:
                self.selected_point = i
                self.status_var.set(f"Selected point {i+1} at ({point_x:.1f}, {point_y:.1f})")
                self.redraw_points()
                return
        
        # If we already have 4 points, don't add more
        if len(self.points) >= 4:
            self.status_var.set("Maximum 4 points allowed. Select a point to move it.")
            return
        
        # Add new point (storing in original image coordinates)
        original_x = canvas_x / self.zoom_factor
        original_y = canvas_y / self.zoom_factor
        self.points.append((original_x, original_y))
        self.selected_point = len(self.points) - 1
        
        self.redraw_points()
        self.status_var.set(f"Added point {len(self.points)} at ({original_x:.1f}, {original_y:.1f})")
    
    def on_key_press(self, event):
        """Handle arrow keys to move selected point"""
        if self.selected_point is None or not self.original_image:
            return
        
        move_amount = 1.0 / self.zoom_factor  # Move by 1 pixel in displayed image
        
        if event.keysym == "Up":
            self.points[self.selected_point] = (
                self.points[self.selected_point][0],
                self.points[self.selected_point][1] - move_amount
            )
        elif event.keysym == "Down":
            self.points[self.selected_point] = (
                self.points[self.selected_point][0],
                self.points[self.selected_point][1] + move_amount
            )
        elif event.keysym == "Left":
            self.points[self.selected_point] = (
                self.points[self.selected_point][0] - move_amount,
                self.points[self.selected_point][1]
            )
        elif event.keysym == "Right":
            self.points[self.selected_point] = (
                self.points[self.selected_point][0] + move_amount,
                self.points[self.selected_point][1]
            )
        else:
            return
        
        self.redraw_points()
        x, y = self.points[self.selected_point]
        self.status_var.set(f"Moved point {self.selected_point+1} to ({x:.1f}, {y:.1f})")
    
    def redraw_points(self):
        """Redraw all points and lines on the canvas"""
        # Clear existing points and lines
        for marker in self.point_markers:
            self.canvas.delete(marker)
        for line in self.lines:
            self.canvas.delete(line)
        
        self.point_markers = []
        self.lines = []
        
        # Draw the points
        for i, (x, y) in enumerate(self.points):
            # Scale point coordinates according to zoom
            scaled_x = x * self.zoom_factor
            scaled_y = y * self.zoom_factor
            
            # Different color for selected point
            color = "red" if i == self.selected_point else "blue"
            size = 6 if i == self.selected_point else 5
            
            # Draw point
            point_id = self.canvas.create_oval(
                scaled_x - size, scaled_y - size,
                scaled_x + size, scaled_y + size,
                fill=color, outline="white"
            )
            self.point_markers.append(point_id)
            
            # Draw point label
            label_id = self.canvas.create_text(
                scaled_x, scaled_y - 15,
                text=str(i+1),
                fill="white", font=("Arial", 9, "bold")
            )
            self.point_markers.append(label_id)
        
        # Draw lines to form polygon
        if len(self.points) > 1:
            for i in range(len(self.points)):
                x1, y1 = self.points[i]
                x2, y2 = self.points[(i + 1) % len(self.points)]  # Connect back to first point
                
                # Scale coordinates according to zoom
                scaled_x1 = x1 * self.zoom_factor
                scaled_y1 = y1 * self.zoom_factor
                scaled_x2 = x2 * self.zoom_factor
                scaled_y2 = y2 * self.zoom_factor
                
                line_id = self.canvas.create_line(
                    scaled_x1, scaled_y1, scaled_x2, scaled_y2,
                    fill="yellow", width=2, dash=(4, 2)
                )
                self.lines.append(line_id)
    
    def reset_points(self):
        """Clear all points"""
        self.points = []
        self.selected_point = None
        if self.original_image:
            self.display_image()
        self.status_var.set("Points reset. Click on the image to add points.")
    
    def save_points(self):
        """Save points to a text file"""
        if len(self.points) != 4:
            messagebox.showerror("Error", "Please create exactly 4 points before saving.")
            return
        
        # Format points according to required schema
        points_data = [[round(x), round(y)] for x, y in self.points]
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Save Point Coordinates"
        )
        
        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(points_data, f)
                self.status_var.set(f"Points saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")

def main():
    root = tk.Tk()
    app = ImagePolygonApp(root)
    root.geometry("800x600")
    root.mainloop()

if __name__ == "__main__":
    main()

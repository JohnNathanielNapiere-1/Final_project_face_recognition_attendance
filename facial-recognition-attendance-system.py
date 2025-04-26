import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
from tkinter import *
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

class FacialRecognitionAttendanceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Recognition Attendance System")
        self.root.geometry("1200x650+0+0")
        
        # Variables
        self.known_face_encodings = []
        self.known_face_names = []
        self.present_students = set()  # Track present students for today
        self.camera_active = False
        self.excel_file_path = "attendance.xlsx"
        
        # Create UI first
        self.create_ui()
        
        # Then load faces and initialize attendance file
        self.load_known_faces()
        self.initialize_attendance_file()
    
    def create_ui(self):
        # Main frames
        title_frame = Frame(self.root, bg="#0077b6", bd=5, relief=RIDGE)
        title_frame.place(x=0, y=0, width=1200, height=70)
        
        title_label = Label(title_frame, text="Facial Recognition Attendance System", 
                          font=("Helvetica", 24, "bold"), bg="#0077b6", fg="white")
        title_label.pack(side=TOP, fill=X)
        
        # Main content frame
        main_frame = Frame(self.root, bd=2, relief=RIDGE, bg="#f8f9fa")
        main_frame.place(x=0, y=70, width=1200, height=580)
        
        # Left frame for camera feed
        self.left_frame = LabelFrame(main_frame, text="Camera Feed", font=("Helvetica", 12), bg="#f8f9fa", bd=2, relief=RIDGE)
        self.left_frame.place(x=10, y=10, width=580, height=550)
        
        self.camera_label = Label(self.left_frame, bg="black")
        self.camera_label.place(x=5, y=5, width=565, height=480)
        
        # Camera control buttons
        btn_start_camera = Button(self.left_frame, text="Start Camera", command=self.start_camera, 
                               font=("Helvetica", 12), bg="#28a745", fg="white", cursor="hand2")
        btn_start_camera.place(x=140, y=490, width=120, height=30)
        
        btn_stop_camera = Button(self.left_frame, text="Stop Camera", command=self.stop_camera, 
                              font=("Helvetica", 12), bg="#dc3545", fg="white", cursor="hand2")
        btn_stop_camera.place(x=280, y=490, width=120, height=30)
        
        # Right frame for attendance display
        right_frame = Frame(main_frame, bd=2, relief=RIDGE, bg="#f8f9fa")
        right_frame.place(x=600, y=10, width=580, height=550)
        
        # Attendance display section
        attendance_frame = LabelFrame(right_frame, text="Attendance Records", font=("Helvetica", 12), bg="#f8f9fa", bd=2, relief=RIDGE)
        attendance_frame.place(x=10, y=10, width=560, height=490)
        
        # Table frame
        table_frame = Frame(attendance_frame, bd=2, relief=RIDGE, bg="white")
        table_frame.place(x=10, y=10, width=535, height=440)
        
        # Scrollbars
        scroll_x = Scrollbar(table_frame, orient=HORIZONTAL)
        scroll_y = Scrollbar(table_frame, orient=VERTICAL)
        
        # Attendance table
        self.attendance_table = ttk.Treeview(
            table_frame,
            columns=("name", "status"),
            xscrollcommand=scroll_x.set,
            yscrollcommand=scroll_y.set
        )
        
        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.attendance_table.xview)
        scroll_y.config(command=self.attendance_table.yview)
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        self.attendance_table.heading("name", text="Student Name")
        self.attendance_table.heading("status", text=f"Status ({today})")
        
        self.attendance_table["show"] = "headings"
        
        self.attendance_table.column("name", width=300)
        self.attendance_table.column("status", width=200)
        
        self.attendance_table.pack(fill=BOTH, expand=1)
        
        # Save button
        btn_save = Button(attendance_frame, text="Save Attendance", command=self.save_attendance, 
                         font=("Helvetica", 12), bg="#17a2b8", fg="white", cursor="hand2")
        btn_save.place(x=200, y=430, width=150, height=30)
        
        # Status bar at bottom
        self.status_bar = Label(right_frame, text="Status: Ready", font=("Helvetica", 10), 
                              bg="#f8f9fa", bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.place(x=10, y=510, width=560, height=25)
        
        # Display initial attendance data
        self.display_attendance()
    
    def format_name(self, filename):
        """Format student name from filename (e.g., 'John_Doe.png' -> 'John Doe')"""
        # Remove file extension
        name = os.path.splitext(filename)[0]
        # Replace underscores with spaces
        name = name.replace('_', ' ')
        # Capitalize each word
        name = ' '.join(word.capitalize() for word in name.split())
        return name
    
    def load_known_faces(self):
        try:
            self.update_status("Loading known faces...")
            # Create "faces" directory if it doesn't exist
            if not os.path.exists("faces"):
                os.makedirs("faces")
                self.update_status("Created 'faces' directory. Please add face images to this folder.")
                return
            
            # Load images from the faces directory
            face_files = [f for f in os.listdir("faces") if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            if not face_files:
                self.update_status("No face images found in 'faces' directory")
                return
                
            for file in face_files:
                # Get formatted name from filename
                name = self.format_name(file)
                
                try:
                    # Load image file
                    face_image_path = os.path.join("faces", file)
                    face_image = face_recognition.load_image_file(face_image_path)
                    
                    # Try to find face encodings
                    face_encoding = face_recognition.face_encodings(face_image)
                    
                    if len(face_encoding) > 0:
                        # Use the first face found in the image
                        self.known_face_encodings.append(face_encoding[0])
                        self.known_face_names.append(name)
                    else:
                        self.update_status(f"No face detected in {file}")
                except Exception as e:
                    self.update_status(f"Error processing {file}: {str(e)}")
            
            self.update_status(f"Loaded {len(self.known_face_names)} known faces")
            
        except Exception as e:
            self.update_status(f"Error loading faces: {str(e)}")
            messagebox.showerror("Error", f"Failed to load faces: {str(e)}")
    
    def clean_and_format_excel(self):
        """Clean and format the Excel file to ensure proper structure"""
        try:
            df = pd.read_excel(self.excel_file_path)
            
            # Step 1: Find the column that contains student names (if it exists)
            name_column = None
            for col in df.columns:
                if any(col.lower().strip() in ['name', 'student', 'student name', 'student_name']):
                    name_column = col
                    break
            
            # If no name column found, look for a column with name-like data
            if name_column is None:
                for col in df.columns:
                    # Check if column values contain common name patterns
                    sample_values = df[col].dropna().astype(str).tolist()[:5]
                    if any(' ' in val for val in sample_values):  # Check if values contain spaces like names do
                        name_column = col
                        break
            
            # If still no name column, create a new DataFrame
            if name_column is None:
                self.update_status("Could not find name column in Excel file. Creating new structure.")
                new_df = pd.DataFrame(columns=['Student Name'])
                return new_df
            
            # Step 2: Extract student names and create a new properly formatted DataFrame
            students = df[name_column].dropna().astype(str).tolist()
            students = [name.strip() for name in students if name.strip()]  # Remove empty strings
            
            # Create a new DataFrame with proper structure
            new_df = pd.DataFrame({'Student Name': students})
            
            # Step 3: Copy over existing attendance data if possible
            date_columns = [col for col in df.columns if col != name_column]
            for date_col in date_columns:
                try:
                    # Try to parse as date to ensure it's a date column
                    if pd.to_datetime(date_col, errors='coerce') is not pd.NaT:
                        # For each student, find their attendance status on this date
                        attendance_dict = {}
                        for _, row in df.iterrows():
                            if pd.notna(row[name_column]) and pd.notna(row[date_col]):
                                student = row[name_column].strip()
                                attendance_dict[student] = row[date_col]
                        
                        # Add the attendance data to the new DataFrame
                        new_df[date_col] = new_df['Student Name'].map(attendance_dict)
                except:
                    # Skip columns that don't contain date information
                    continue
            
            # Step 4: Sort alphabetically by student name
            new_df = new_df.sort_values('Student Name').reset_index(drop=True)
            
            self.update_status("Excel file cleaned and properly formatted")
            return new_df
            
        except Exception as e:
            self.update_status(f"Error cleaning Excel file: {str(e)}")
            # Return empty DataFrame if cleaning fails
            return pd.DataFrame(columns=['Student Name'])
    
    def initialize_attendance_file(self):
        try:
            # Get list of students from image files
            students_from_images = sorted(self.known_face_names)
            
            # Check if excel file exists
            if os.path.exists(self.excel_file_path):
                try:
                    # First try to read it as is
                    df = pd.read_excel(self.excel_file_path)
                    
                    # Check if the file has the correct structure
                    if 'Student Name' not in df.columns:
                        # If not properly formatted, clean and standardize it
                        self.update_status("Excel file not properly formatted. Cleaning and reformatting...")
                        df = self.clean_and_format_excel()
                        
                    # Get list of students from Excel file
                    students_from_excel = df['Student Name'].tolist()
                    
                    # Check for new students (in images but not in Excel)
                    new_students = [s for s in students_from_images if s not in students_from_excel]
                    
                    if new_students:
                        self.update_status(f"Adding {len(new_students)} new students to attendance sheet")
                        
                        # Add new students to DataFrame
                        for student in new_students:
                            new_row = pd.DataFrame({'Student Name': [student]})
                            df = pd.concat([df, new_row], ignore_index=True)
                        
                        # For each date column except today, mark new students as "Not Enrolled Yet"
                        today = datetime.now().strftime("%Y-%m-%d")
                        date_columns = [col for col in df.columns if col != 'Student Name']
                        
                        for col in date_columns:
                            if col != today:
                                for student in new_students:
                                    df.loc[df['Student Name'] == student, col] = "Not Enrolled Yet"
                    
                    # Make sure today's column exists
                    today = datetime.now().strftime("%Y-%m-%d")
                    if today not in df.columns:
                        df[today] = "Absent"
                    
                    # Sort by student name
                    df = df.sort_values('Student Name').reset_index(drop=True)
                    
                    # Save the updated DataFrame
                    df.to_excel(self.excel_file_path, index=False)
                    
                    # Initialize the set of present students for today
                    present_students = df[df[today] == "Present"]['Student Name'].tolist()
                    self.present_students = set(present_students)
                    
                    self.update_status("Attendance file updated successfully")
                    
                except Exception as e:
                    self.update_status(f"Error reading Excel file: {str(e)}. Creating new file.")
                    # If error occurs during reading, create a new file
                    self.create_new_attendance_file(students_from_images)
            else:
                # Create new attendance file
                self.create_new_attendance_file(students_from_images)
            
        except Exception as e:
            self.update_status(f"Error initializing attendance file: {str(e)}")
            messagebox.showerror("Error", f"Failed to initialize attendance file: {str(e)}")
    
    def create_new_attendance_file(self, students):
        """Create a new attendance file with the given student list"""
        # Create a new DataFrame with student names
        df = pd.DataFrame({'Student Name': students})
        
        # Add today's date column with all students marked as 'Absent'
        today = datetime.now().strftime("%Y-%m-%d")
        df[today] = "Absent"
        
        # Sort by student name
        df = df.sort_values('Student Name').reset_index(drop=True)
        
        # Save the DataFrame to Excel
        df.to_excel(self.excel_file_path, index=False)
        self.update_status(f"Created new attendance file: {self.excel_file_path}")
    
    def display_attendance(self):
        try:
            # Clear existing items in the table
            self.attendance_table.delete(*self.attendance_table.get_children())
            
            # Read the attendance data
            if os.path.exists(self.excel_file_path):
                df = pd.read_excel(self.excel_file_path)
                today = datetime.now().strftime("%Y-%m-%d")
                
                # Check if today's column exists
                if today not in df.columns:
                    df[today] = "Absent"
                    df.to_excel(self.excel_file_path, index=False)
                
                # Populate the table with attendance records
                for _, row in df.iterrows():
                    self.attendance_table.insert("", END, values=(row["Student Name"], row[today]))
                
                self.update_status(f"Displayed attendance records for {len(df)} students")
            else:
                self.update_status("No attendance records found")
                
        except Exception as e:
            self.update_status(f"Error displaying attendance: {str(e)}")
            messagebox.showerror("Error", f"Failed to display attendance: {str(e)}")
    
    def mark_attendance(self, name):
        try:
            # If already marked present, skip
            if name in self.present_students:
                return False
            
            # Add to set of present students
            self.present_students.add(name)
            
            # Read the current Excel file
            df = pd.read_excel(self.excel_file_path)
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Update the attendance status for the recognized student
            df.loc[df['Student Name'] == name, today] = "Present"
            
            # Save the updated DataFrame
            df.to_excel(self.excel_file_path, index=False)
            
            # Refresh the display
            self.display_attendance()
            
            self.update_status(f"Marked {name} as present")
            return True
            
        except Exception as e:
            self.update_status(f"Error marking attendance: {str(e)}")
            messagebox.showerror("Error", f"Failed to mark attendance: {str(e)}")
            return False
    
    def save_attendance(self):
        try:
            if os.path.exists(self.excel_file_path):
                # Make a backup of the current attendance file
                backup_path = f"attendance_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                df = pd.read_excel(self.excel_file_path)
                df.to_excel(backup_path, index=False)
                
                self.update_status(f"Attendance saved. Backup created: {backup_path}")
                messagebox.showinfo("Save Successful", "Attendance data has been saved successfully")
            else:
                self.update_status("No attendance data to save")
                messagebox.showwarning("Save Failed", "No attendance data found")
                
        except Exception as e:
            self.update_status(f"Error saving attendance: {str(e)}")
            messagebox.showerror("Error", f"Failed to save attendance: {str(e)}")
    
    def update_status(self, message):
        try:
            if hasattr(self, 'status_bar'):
                self.status_bar.config(text=f"Status: {message}")
            print(message)  # Always log to console for debugging
        except Exception as e:
            print(f"Failed to update status: {message} - Error: {str(e)}")
    
    def start_camera(self):
        if not self.camera_active:
            self.camera_active = True
            self.update_status("Starting camera...")
            
            # Start camera in a separate thread
            threading.Thread(target=self.video_stream, daemon=True).start()
    
    def stop_camera(self):
        self.camera_active = False
        self.update_status("Camera stopped")
    
    def video_stream(self):
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        process_this_frame = True
        
        while self.camera_active:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                self.update_status("Failed to capture frame from camera")
                break
                
            # For performance, process every other frame
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                
                # Convert from BGR to RGB
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in the frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                
                for face_encoding in face_encodings:
                    # Recognition mode - try to match with known faces
                    if len(self.known_face_encodings) > 0:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"
                        
                        # Use the known face with the smallest distance
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = self.known_face_names[best_match_index]
                                # Mark attendance for recognized person
                                if name != "Unknown":
                                    self.mark_attendance(name)
                        
                        face_names.append(name)
                
                # Draw results on the frame
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            process_this_frame = not process_this_frame
            
            # Convert to PhotoImage for display
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the camera label
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)
            
        # Release the camera when done
        cap.release()
        # Clear the camera label
        self.camera_label.configure(image="")

if __name__ == "__main__":
    root = Tk()
    app = FacialRecognitionAttendanceSystem(root)
    root.mainloop()

import customtkinter
# This is a UI-Library
from tkinter import filedialog
#from PIL import Image
#PIL is a legacy package replaced by pillow
from pillow import Image

import pandas as pd 
import numpy as np
# import tensorflow as tf
from tensorflow import lite
# this imports the ML library
import cv2
# this is a computer vision library 
from pathlib import Path
from datetime import datetime

# import classification

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Constants
# Define the weight of each level
WEIGHTS = [1,2,3,4,5]
# Define the mappings for each level
MAPPINGS = [2,0,1,4,3]
# Define the model file
MODEL = "full_model_5_levels.tflite"

# class Frame1(customtkinter.CTkFrame):
#     def __init__(self, master):
#         super().__init__(master)

#         #if you remove this, the "red block" will stick to the upper left corner
#         master.grid_columnconfigure(0, weight=1)
#         master.grid_rowconfigure(0, weight=1)
        
#         self.frame_rot = customtkinter.CTkFrame(master=master, fg_color='red')
#         self.frame_rot.grid(row=0, column=0)

#         #sticky "ns" will centralize the label vertically
#         self.label = customtkinter.CTkLabel(master=self.frame_rot, text='hello', height=130)
#         # self.label.grid(row=0, column=0, sticky="ns")
#         self.label.grid(row=0, column=0)



# class Frame1(customtkinter.CTkFrame):
#     def __init__(self, master):
#         super().__init__(master)

# CTK GUI class
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{1200}x{600}")
        self.minsize(width=600, height=400)
        self.title("Test of Creative Thinking - Drawing Production (TCT-DP)")
        # self.maxsize(width=1600, height=1200)

        # configure grid system
        # self.grid_rowconfigure((1,2,3), weight=1)  
        # self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(2, weight=1)
        # self.grid_columnconfigure

        # Initialize the image path variable
        self.image_path = "./example images/12102021111421-0012.jpg"
        # Initialize the image folder path variable
        self.image_folder_path = ""
        # Initialize the image object
        self.single_image = Image.open(self.image_path)

        # Initialize the results variable as an empty numpy array
        self.class_results = np.empty(0)
        # Initialize another results variable to store a previous version of the results
        self.class_results_prev = np.empty(0)

        # Initialize the a flag variable to check if classifications is running
        self.is_running = False

        # Set the title font
        self.title_font = customtkinter.CTkFont(size=20, weight="bold")
         # Set the label title font
        self.label_title_font = customtkinter.CTkFont(size=14, weight="bold")
        # Set the textbox font
        self.textbox_font = customtkinter.CTkFont(size=15)
        #Set bold font
        bold = customtkinter.CTkFont(weight="bold")

        ####################################################################
        # Sidebar menu
        # self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        # self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nswe")
        # self.sidebar_frame.grid_rowconfigure((4,8), weight=1)

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row = 0, column = 0, rowspan=6, sticky="nswe")
        # self.sidebar_frame.grid_rowconfigure((4,8), weight=1)

        #Sidebar content
        row_index = 0

        # Title
        self.title_label = customtkinter.CTkLabel(self.sidebar_frame, text="Side Menu", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.title_label.grid(row = row_index, column=0, padx=20, pady=(20, 10), sticky="nw")
        row_index+=1

        # Checkbox to crop image
        self.checkbox_1 = customtkinter.CTkCheckBox(self.sidebar_frame,text="Crop image", font=bold, command=self.crop_message)
        self.checkbox_1.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        row_index+=1
        # Button to choose single image
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Select Single Image", font=bold, command=self.get_image_path)
        self.sidebar_button_1.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        row_index+=1
        # self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Save Result", font=bold,  command= save_results)
        # self.sidebar_button_1.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        # row_index+=1
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Select Image Folder", font=bold, command=self.get_image_folder_path)
        # self.sidebar_button_2.configure(text="Select Image Folder")
        self.sidebar_button_2.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        row_index+=1
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Start Batch Classification", font=bold, command=self.batch_classification)
        self.sidebar_button_3.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        row_index+=1
        # self.sidebar_frame.grid_rowconfigure(row_index, weight=1)
        row_index+=1
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Save Result", font=bold, text_color="yellow", command=self.save_results)
        self.sidebar_button_4.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        row_index+=1
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Save As", font=bold, text_color="yellow", command=self.save_as)
        self.sidebar_button_5.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        row_index+=1
        self.sidebar_frame.grid_rowconfigure(row_index, weight=1)
        row_index+=1

        # Button to clear contents in message window
        self.sidebar_button_6 = customtkinter.CTkButton(self.sidebar_frame, text="Clear Message", font=bold, command=self.clear_message)
        self.sidebar_button_6.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        row_index+=1

        # Checkbox to show/hide description
        self.checkbox_2 = customtkinter.CTkCheckBox(self.sidebar_frame,text="Show/Hide\n Description", font=bold, command=self.toggle_description)
        self.checkbox_2.grid(row = row_index, column=0, padx=20, pady=10, sticky="nw")
        row_index+=1
        # Appearance mode selection
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", font=bold, anchor="w")
        self.appearance_mode_label.grid(row = row_index, column=0, padx=20, pady=(10, 0), sticky="nw")
        row_index+=1
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row = row_index, column=0, padx=20, pady=(10, 10), sticky="nw")
        row_index+=1
        # UI scaling selection
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", font=bold, anchor="w")
        self.scaling_label.grid(row = row_index, column=0, padx=20, pady=(10, 0), sticky="nw")
        row_index+=1
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row = row_index, column=0, padx=20, pady=(10, 20), sticky="nw")


        # default appearance mode is light
        self.appearance_mode_optionemenu.set("Light")
        #default UI scaling is 100%
        self.scaling_optionemenu.set("100%")

        ####################################################################
        # Information frame
        # Generate textbox to display information
        
        self.info_row_index = 0

        # Message window title
        self.title_label = customtkinter.CTkLabel(self, text="Message", font=self.title_font)
        self.title_label.grid(row = self.info_row_index, column=2, padx=20, pady=(20, 10), sticky="nw")
        self.info_row_index+=1
        # Text box for event message
        self.message_box = customtkinter.CTkTextbox(self, width=50*6, font=self.textbox_font, border_width=2, wrap = "word")
        self.message_box.grid(row = self.info_row_index, column=2, padx=20, pady=(0, 10), sticky="nsew")
        # row_index+=1
        self.grid_rowconfigure(self.info_row_index, weight=1)
        self.info_row_index+=1

        self.message_button_1 = customtkinter.CTkButton(self, text="Clear Message", font=bold, command=self.clear_message)
        self.message_button_1.grid(row = self.info_row_index, column=2, padx=20, pady=10, sticky="nw")
        self.info_row_index+=1

        # Create a frame for description
        self.description_frame = customtkinter.CTkFrame(self, fg_color="transparent", corner_radius=0)
        self.description_frame.grid(row = self.info_row_index, column = 2, rowspan=4, sticky="nswe")
        self.description_frame.columnconfigure(0,weight=1)
        # self.description_frame.rowconfigure((1,2,3),weight=1)

        #Title for program description
        self.title_label_description = customtkinter.CTkLabel(self.description_frame, text="Description", font=self.title_font)
        self.title_label_description.grid(row = 0, column=0, padx=20, pady=(20, 10), sticky="nw")

        # self.info_row_index+=1
        # info_filenames = Path("./info/")
        for row, filename in enumerate(Path("./info/").glob("*.txt")):
            self.textbox = customtkinter.CTkTextbox(self.description_frame, width=50*6, fg_color="transparent", font=self.textbox_font, border_width=2, wrap = "word")
            self.textbox.grid(row=row+1, column=0, padx=20, pady=(0, 20), sticky="nsew")

            # print(filename)

            # Open and read the file
            with open(filename, "r") as f:
                # info_title = f.readline()
                # info_description = f.read()
                info = f.read()
            # Add the content to the textbox
            self.textbox.insert("0.0", info)
        
        # Default value for checkbox2, showing description
        self.checkbox_2.select()


        ####################################################################
        # Image frame for classification result and image display
        self.image_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.image_frame.grid(row=0, column=1, rowspan=6, sticky="nswe")
        # self.image_frame.grid_rowconfigure(1, weight=1)
        # self.image_frame.grid_rowconfigure((2,3,4,5), weight=0)

        # Title for table
        self.title_label = customtkinter.CTkLabel(self.image_frame, text="Classification Results", font=self.title_font)
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nw")

        #Table frame
        self.result_table = customtkinter.CTkFrame(self.image_frame,border_width=2)
        self.result_table.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        # table values
        val_table = [["Category", "1", "2", "3", "4", "5"],
                     ["Label", "Far Below Average", "Below Average", 
                      "Average", "Above Average", "Far Above Average"],
                     ["Confidence Score (%)", "0", "0", "0", "0", "0"]]
        
        self.confidence_table = []
        # generate table cells
        for i in range(len(val_table)):
            for j in range(len(val_table[i])):
                cell = customtkinter.CTkLabel(self.result_table, text=val_table[i][j], font=customtkinter.CTkFont(size=14, weight="bold"))
                cell.grid(row=i, column=j, padx=8, pady=5)
                if i==2 and j >0:
                    self.confidence_table.append(cell)

        cell = customtkinter.CTkLabel(self.result_table, text="Weighted Score",font=customtkinter.CTkFont(size=14, weight="bold"))
        cell.grid(row=4, column=0, padx=8, pady=5)
        self.cell_weighted_score = customtkinter.CTkLabel(self.result_table, text="score",font=customtkinter.CTkFont(size=14, weight="bold"))
        self.cell_weighted_score.grid(row=4, column=3, padx=8, pady=5)

        # Create a frame for the last/next buttons
        self.button_frame = customtkinter.CTkFrame(self.image_frame, fg_color="transparent", corner_radius=0)
        self.button_frame.grid(row = 2, column = 0, padx=20, pady=(30,0), sticky="nswe")
        # self.button_frame.columnconfigure(0,weight=1)
        self.last_button = customtkinter.CTkButton(self.button_frame, text="Last", font=bold, command=self.get_last_image)
        self.last_button.grid(row = 0, column=0, sticky="nw")
        self.next_button = customtkinter.CTkButton(self.button_frame, text="Next", font=bold, command=self.get_next_image)
        self.next_button.grid(row = 0, column=1, padx=(5,0), sticky="nw")

        # Show image
        self.show_single_image()
        self.add_message("Displaying example image: {}".format(Path(self.image_path).name))
        # Show image path
        self.image_tile = customtkinter.CTkLabel(self.image_frame, text= "Image: " + self.image_path, font=customtkinter.CTkFont(size=15))
        self.image_tile.grid(row=3, column=0, padx=20, sticky="nw")


        # Get a list of image paths in the same folder
        parent_folder_path = str(Path(self.image_path).parents[0])
        self.image_paths = [path for path in Path(parent_folder_path).iterdir() if path.suffix in {".jpg", ".png", ".bmp"}]
        print("Image paths: {}".format(self.image_paths))
        print(self.image_paths.index(Path(self.image_path)))


        
    ####################################################################
    # Functions

    # Function to change appearance mode
    def change_appearance_mode_event(self):
        self.appearance_mode = self.appearance_mode_optionemenu.get()   
    
    # Function to get image path
    def get_image_path(self):
        image_path = filedialog.askopenfilename(initialdir=Path().resolve(), title="Select an image file",  
                                                filetypes=(("jpg files", "*.jpg"),("png files", "*.png"),("bmp files", "*.bmp")))
        
        if image_path == "":
            self.add_message("Warning: Image is not selected, please select an image first!")
            return
        
        self.image_path = image_path
        self.show_single_image()

        # Get a list of image paths in the same folder
        parent_folder_path = str(Path(self.image_path).parents[0])
        self.image_paths = [path for path in Path(parent_folder_path).iterdir() if path.suffix in {".jpg", ".png", ".bmp"}]
        print("Image paths: {}".format(self.image_paths))
        print(self.image_paths.index(Path(self.image_path)))
        

    # Function to choose and show single image
    def show_single_image(self):

        try:
            self.single_image = Image.open(self.image_path)
            # print(self.single_image.size)
            width, height = self.single_image.size
            # Crop image if checkbox is checked
            if self.checkbox_1.get():
                self.single_image = self.single_image.crop((0, height*0.2, width, height*0.8)) 
                self.add_message("Orginal Size:{}X{}\nCropped Size{}X{}".format(
                    width, height, self.single_image.size[0], self.single_image.size[1]))
            # image = image.resize(400*400)
            # print(self.single_image.size)
            self.add_message("Displaying selected image: {} ".format(Path(self.image_path).name))
            self.single_classification()
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print (message)
        else:

        # if self.image_path != "":
        #     self.single_image = Image.open(self.image_path)
        #     # print(self.single_image.size)
        #     width, height = self.single_image.size
        #     # Crop image if checkbox is checked
        #     if self.checkbox_1.get():
        #         self.single_image = self.single_image.crop((0, height*0.2, width, height*0.8)) 
        #         self.add_message("Orginal Size:{}X{}\nCropped Size{}X{}".format(
        #             width, height, self.single_image.size[0], self.single_image.size[1]))
        #     # image = image.resize(400*400)
        #     # print(self.single_image.size)
        #     self.add_message("Displaying selected image: {} ".format(Path(self.image_path).name))
        #     self.single_classification()
        # else:   
        #     # Display example image if not already selected
        #     self.image_path = str(Path("./example images/12102021111421-0012.jpg").resolve())

        #     self.single_image = Image.open(self.image_path)
        
            self.single_image = customtkinter.CTkImage(light_image=self.single_image,
                    dark_image=self.single_image,
                    size=(800, 800))
            self.image_label = customtkinter.CTkLabel(self.image_frame, image=self.single_image, text=None)  # display image with a CTkLabel
            self.image_label.grid(row=4, column=0, padx=20, pady=(0,10), sticky = "nswe", rowspan=3)

    # Function to change appearance mode
    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")


    # Function to add message to message box
    def add_message(self, message: str):
        # Enable message box input
        self.message_box.configure(state="normal")
        # insert message
        self.message_box.insert("0.0", message + "\n")
        # Disable message box input
        self.message_box.configure(state="disabled")

    # Function to clear message from message box
    def clear_message(self):
        # try:
        #     self.message_box.configure(state="normal")
        #     # if self.message_box.cget(text)
        #     self.message_box.delete("0.0","end")
        #     self.message_box.configure(state="disabled")
        # except Exception as ex:
        #     template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        #     message = template.format(type(ex).__name__, ex.args)
        #     print (message)
        # else:
        #     print("error clearing")
        self.message_box.configure(state="normal")
        # if self.message_box.cget(text)
        self.message_box.delete("0.0","end")
        self.message_box.configure(state="disabled")
    
    # Function to show/hide description
    def toggle_description(self):
        if self.checkbox_2.get():
            self.description_frame.grid()
            self.add_message("Show description")
        else:
            self.description_frame.grid_remove()
            self.add_message("Hide description")
            
    def crop_message(self):
        if self.checkbox_1.get():
            self.add_message("Crop Image Checked")
        else:
            self.add_message("Crop Image Unchecked")
    
    # Function to get image folder path
    def get_image_folder_path(self):
        if self.is_running:
            self.add_message("Warning: Batch classification is running!")
            return
        
        self.image_folder_path = filedialog.askdirectory(initialdir=Path().resolve())
        if self.image_folder_path == "":
            self.add_message("Warning: Image folder is not selected, please select an image folder first!")
        else:
            self.add_message("Selected image folder path: {}".format(self.image_folder_path))

    # Function to get last image in the same folder
    def get_last_image(self):
        try:
            index = self.image_paths.index(Path(self.image_path))
            if index == 0:
                self.add_message("First image in folder")
            else:
                self.image_path = str(self.image_paths[index-1])
                self.update_image()
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.add_message(message)
            print (message)
        else:
            print(self.image_path)
            # self.update_image()
    
    # Function to get next image in the same folder
    def get_next_image(self):
        try:
            index = self.image_paths.index(Path(self.image_path))
            self.image_path = str(self.image_paths[index+1])
        except IndexError:
            self.add_message("Last image in folder")
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            self.add_message(message)
            print (message)
        else:
            print(self.image_path)
            self.update_image()


    # Function to update the shown image and path
    def update_image(self):
        self.image_tile.configure(text= "Image: " + self.image_path)
        self.show_single_image()

    # Function to save classification results
    def save_results(self):
        # creating a list of column names 
        # column_values = ['Filename','Output Data', 'Output Data','Confidence Score', 'Confidence Score','Weighted Score']
        if self.is_running:
            self.add_message("Warning: Batch classification is running!")
            return 
        if self.class_results.size == 0:
            self.add_message("No results to save!")
            return
        if np.array_equal(self.class_results, self.class_results_prev):
            self.add_message("No changes to save!")
            return

        self.class_results_prev = self.class_results.copy()
        column1 = ['', 'Output Data', 'Output Data','Output Data','Output Data','Output Data',
                   'Confidence Score','Confidence Score','Confidence Score','Confidence Score', 'Confidence Score', '']
        column2 = ['Filename', 'Far Below Average', 'Below Average', 'Average', 'Above Average', 'Far Above Average',
                    'Far Below Average', 'Below Average', 'Average', 'Above Average', 'Far Above Average', 'Weighted Score']
        column_values = pd.MultiIndex.from_arrays([column1,column2])

        # creating the dataframe 
        df = pd.DataFrame(data = self.class_results,  columns = column_values) 
        # print(df) 

        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
        save_path = str(Path("./results/"+ dt_string + ".csv").resolve())
        # print(save_path)
        df.to_csv(save_path)
        self.add_message("Classification results saved to " + save_path)

    # Function to save classification results automatically
    def auto_save(self, batch = False):

        column1 = ['', 'Output Data', 'Output Data','Output Data','Output Data','Output Data',
                'Confidence Score','Confidence Score','Confidence Score','Confidence Score', 'Confidence Score', '']
        column2 = ['Filename', 'Far Below Average', 'Below Average', 'Average', 'Above Average', 'Far Above Average',
                'Far Below Average', 'Below Average', 'Average', 'Above Average', 'Far Above Average', 'Weighted Score']
        column_values = pd.MultiIndex.from_arrays([column1,column2])

        # column_values = ['Filename','Output Data', 'Output Data','Output Data','Output Data','Output Data',
        #                  'Confidence Score', 'Confidence Score','Confidence Score','Confidence Score','Confidence Score','Weighted Score']
        # creating the dataframe 
        df = pd.DataFrame(data = self.class_results,  columns = column_values) 
        # print(df)
        if batch:
            df.to_csv('batch_autosave_results.csv')
            self.add_message("Classification results saved to batch_autosave_results.csv")
        else:
            df.to_csv('autosave_results.csv')
            self.add_message("Classification results saved to autosave_results.csv")

    # Function to save classification results in the specified directory
    def save_as(self):
        if self.is_running:
            self.add_message("Warning: Batch classification is running!")
            return
        if self.class_results.size == 0:
            self.add_message("No results to save!")
            return
        
        save_path = filedialog.asksaveasfilename(initialdir=Path().resolve(), defaultextension = ".csv", filetypes=(("csv files", "*.csv"),("all files", "*.*")))
        if save_path == "":
            self.add_message("Warning: File path is not selected, please select a file path first!")
            return
        
        column1 = ['', 'Output Data', 'Output Data','Output Data','Output Data','Output Data',
                   'Confidence Score','Confidence Score','Confidence Score','Confidence Score', 'Confidence Score', '']
        column2 = ['Filename', 'Far Below Average', 'Below Average', 'Average', 'Above Average', 'Far Above Average',
                    'Far Below Average', 'Below Average', 'Average', 'Above Average', 'Far Above Average', 'Weighted Score']
        column_values = pd.MultiIndex.from_arrays([column1,column2])

        # column_values = ['Filename','Output Data', 'Output Data','Output Data','Output Data','Output Data',
        #                  'Confidence Score', 'Confidence Score','Confidence Score','Confidence Score','Confidence Score','Weighted Score']
        # creating the dataframe 
        df = pd.DataFrame(data = self.class_results,  columns = column_values) 
        # print(df) 
        df.to_csv(save_path)
        self.add_message("Classification results saved to " + save_path)

    # Function to classify single image
    def single_classification(self):
        if self.is_running:
            self.add_message("Warning: Batch classification is running!")
            return
        if self.image_path == "":
            self.add_message("Warning: Image is not selected, please select an image first!")
            return
        
        self.add_message("Single classification started...")
        # Set the flag to true when the classification is running
        self.is_running = True


        # Get the file path of the model
        model_path = str(Path("./model/"+ MODEL).resolve())
        # model_path = Path(model_name).resolve()
        # print(model_path)

        # Load the TFLite model and allocate tensors.
        # interpreter = tf.lite.Interpreter(model_path)
        interpreter = lite.Interpreter(model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        img = cv2.imread(self.image_path)
        height, width, *_ = img.shape
        if self.checkbox_1.get():
            img = img[int(height*0.2):int(height*0.8), 0:width]
        new_img = cv2.resize(img, (224, 224))
        # input_details[0]['index'] = the index which accepts the input
        interpreter.set_tensor(input_details[0]['index'], [new_img])
        # run the inference
        interpreter.invoke()
        # output_details[0]['index'] = the index which provides the input
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # Squeeze the output data to a 1D array
        output_data = np.squeeze(output_data)
        # Reorder the output based on the label mapping
        output_data = output_data[MAPPINGS]
        # calculate the confidence score using the output data
        confidence_score = output_data/output_data.sum()*100
        # calculate the weighted score using the confidence score
        weighted_score = np.dot(WEIGHTS, confidence_score/100)
        self.add_message("Image processed: {}, Output Data: {} Confinence Score: {}, Weighted Score: {}".
                            format(Path(self.image_path).name, output_data, confidence_score.round(2), weighted_score.round(2)))

        # Update weighted score
        self.cell_weighted_score.configure(text=str((weighted_score).round(2)))

        # Update confidence score

        # self.confidence_table[0].configure(text=str(confidence_score[0].round(2)))
        # self.confidence_table[2].configure(text=str(confidence_score[1].round(2)))
        # For full model
        for i in range(len(confidence_score)):
            self.confidence_table[i].configure(text=str(confidence_score[i].round(2)))


        # creating the Numpy array 
        results = []
        results = np.append(results,Path(self.image_path).name)
        results = np.append(results,output_data)
        results = np.append(results,confidence_score)
        results = np.append(results,weighted_score)
        self.class_results = np.reshape(results,(1,results.size))

        # Reset the flag to false after saving
        self.is_running = False

        # Save the results
        self.auto_save()
    

    
    # Fucntion to classify images
    def batch_classification(self):
        if self.is_running:
            self.add_message("Warning: Batch classification is running!")
            return
        if self.image_folder_path == "":
            self.add_message("Warning: Image folder is not selected, please select an image folder first!")
            return
        
        self.add_message("Batch classification started...")

        # Set the flag to true when the classification is running
        self.is_running = True

        # Get the file path of the model
        model_path = Path("./model/"+ MODEL).resolve()
        # model_path = Path(model_name).resolve()
        # print(model_path)

        # Load the TFLite model and allocate tensors.
        # interpreter = tf.lite.Interpreter(model_path)
        interpreter = lite.Interpreter(str(model_path))
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Print the model's input and output tensors.
        # print(input_details)
        # print(output_details)

        # Empty arrays for the results
        outputs = []
        confidences = []
        scores = []
        filenames = []

        for file in Path(self.image_folder_path).iterdir():         
            # read and resize the image
            img = cv2.imread(r"{}".format(file.resolve()))
            height, width, *_ = img.shape
            if self.checkbox_1.get():
                img = img[int(height*0.2):int(height*0.8), 0:width]
                self.add_message("Image cropped")
                
            new_img = cv2.resize(img, (224, 224))
            # cv2.imshow("Image",new_img)

            # input_details[0]['index'] = the index which accepts the input
            interpreter.set_tensor(input_details[0]['index'], [new_img])

            # run the inference
            interpreter.invoke()

            # output_details[0]['index'] = the index which provides the input
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Squeeze the output data to a 1D array
            output_data = np.squeeze(output_data)
            # Reorder the output based on the label mapping
            output_data = output_data[MAPPINGS]

            # calculate the confidence score using the output data
            confidence_score = output_data/output_data.sum()*100
            # calculate the weighted score using the confidence score
            weighted_score = np.dot(WEIGHTS, confidence_score/100)

            self.add_message("Image processed: {}, Output Data: {} Confinence Score: {}, Weighted Score: {}".
                            format(file.name, output_data, confidence_score.round(2), weighted_score.round(2)))

            np1 = np.squeeze(confidence_score)
            np2 = np.squeeze(weighted_score)
            filenames = np.append(filenames, file.name)

            outputs = np.append(outputs, output_data)
            confidences = np.append(confidences, np1)
            scores = np.append(scores, np2)

        # Reshape the arrays
        outputs = np.reshape(outputs,(filenames.size,len(WEIGHTS)))
        confidences = np.reshape(confidences,(filenames.size,len(WEIGHTS)))
        scores = np.reshape(scores,(filenames.size,1 ))
        filenames = np.reshape(filenames,(filenames.size,1 ))

        # Concatenate the arrays
        self.class_results = np.concatenate((filenames,outputs, confidences, scores), axis=1)

        # Reset the flag to false after saving
        self.is_running = False

        # Save the results
        self.auto_save(batch=True)

if __name__ == "__main__":
    # column1 = ['', 'Output Data', 'Output Data','Confidence Score', 'Confidence Score', '']
    # column2 = ['Filename', 'FBA', 'AVE', 'FBA', 'AVE', 'Weighted Score']
    # column_values = pd.MultiIndex.from_arrays([column1,column2])

    # # column_values = ['Filename','Output Data', 'Output Data','Output Data','Output Data','Output Data',
    # #                  'Confidence Score', 'Confidence Score','Confidence Score','Confidence Score','Confidence Score','Weighted Score']
    # # creating the dataframe 
    # df = pd.DataFrame(columns = column_values)

    app = App()
    app.mainloop()

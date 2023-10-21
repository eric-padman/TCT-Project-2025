import customtkinter
from tkinter import filedialog
from PIL import Image
import os

import pandas as pd 
import numpy as np
import tensorflow as tf
import cv2
import pathlib
from pathlib import Path
import math
from datetime import datetime

# import classification

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

# Constant variables
# Define the weight of each level
WEIGHTS = [1,3]


# CTK GUI class
class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry(f"{1200}x{600}")
        self.minsize(width=600, height=400)
        self.title("Test of Creative Thinking - Drawing Production (TCT-DP)")
        # self.maxsize(width=1600, height=1200)

        # configure grid system
        self.grid_rowconfigure((1,2,3), weight=1)  
        self.grid_columnconfigure(1, weight=1)

        # Initialize the image path variable
        self.image_path = ""
        # Initialize the image folder path variable
        self.image_folder_path = ""
        # Initialize the image object
        self.single_image = None

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

        ####################################################################
        # Sidebar menu
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nswe")
        self.sidebar_frame.grid_rowconfigure((4,8), weight=1)

        #Sidebar content
        # Title
        self.title_label = customtkinter.CTkLabel(self.sidebar_frame, text="Side Menu", font=self.title_font)
        self.title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nw")

        # Checkbox to crop image
        self.checkbox_1 = customtkinter.CTkCheckBox(self.sidebar_frame,text="Crop image", font=self.label_title_font, command=self.checkbox_event)
        self.checkbox_1.grid(row=1, column=0, padx=20, pady=10, sticky="nw")
        
        # Button to choose single image
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Select Single Image", command=self.get_image_path)
        self.sidebar_button_1.grid(row=2, column=0, padx=20, pady=10, sticky="nw")
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, text="Save Result", font =self.label_title_font,  command=self.save_results)
        self.sidebar_button_1.grid(row=3, column=0, padx=20, pady=10, sticky="nw")
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, text="Select Image Folder", command=self.get_image_folder_path)
        self.sidebar_button_2.grid(row=5, column=0, padx=20, pady=10, sticky="nw")
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, text="Start Batch Classification", command=self.batch_classification)
        self.sidebar_button_3.grid(row=6, column=0, padx=20, pady=10, sticky="nw")
        self.sidebar_button_4 = customtkinter.CTkButton(self.sidebar_frame, text="Save Result", font =self.label_title_font, command=self.save_results)
        self.sidebar_button_4.grid(row=7, column=0, padx=20, pady=10, sticky="nw")
        self.sidebar_button_5 = customtkinter.CTkButton(self.sidebar_frame, text="Save As", font =self.label_title_font, command=self.save_as)
        self.sidebar_button_5.grid(row=9, column=0, padx=20, pady=10, sticky="nw")


        # Appearance mode selection
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", font=self.label_title_font, anchor="w")
        self.appearance_mode_label.grid(row=10, column=0, padx=20, pady=(10, 0), sticky="nw")
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 10), sticky="nw")

        # UI scaling selection
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", font=self.label_title_font, anchor="w")
        self.scaling_label.grid(row=12, column=0, padx=20, pady=(10, 0), sticky="nw")
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=13, column=0, padx=20, pady=(10, 20), sticky="nw")

        # default appearance mode is dark
        self.appearance_mode_optionemenu.set("Dark")
        #default UI scaling is 100%
        self.scaling_optionemenu.set("100%")

        ####################################################################
        # Information frame
        #Title for program description
        self.title_label = customtkinter.CTkLabel(self, text="Description", font=self.title_font)
        self.title_label.grid(row=0, column=1, padx=20, pady=(20, 10), sticky="nw")

        # Generate textbox to display information
        info_filenames = ['info1.txt', 'info2.txt', 'Function Description.txt']
        for row, filename in enumerate(info_filenames):
            self.textbox = customtkinter.CTkTextbox(self, width=50*6, fg_color="transparent", font=self.textbox_font, border_width=2, wrap = "word")
            self.textbox.grid(row=row+1, column=1, padx=20, pady=(20, 0), sticky="nsew")

            # Load the description file
            filepath = os.path.join(os.getcwd(), 'info', filename)
            filepath = os.path.expanduser(filepath)
            # print(filepath)

            # Open and read the file
            with open(filepath, "r") as f:
                # info_title = f.readline()
                # info_description = f.read()
                info = f.read()
            # Add the content to the textbox
            self.textbox.insert("0.0", info)
            self.textbox.configure(state="disabled")

        # Message window title
        self.title_label = customtkinter.CTkLabel(self, text="Message", font=self.title_font)
        self.title_label.grid(row=4, column=1, padx=20, pady=(20, 10), sticky="nw")
        # Text box for event message
        self.message_box = customtkinter.CTkTextbox(self, width=50*6, font=self.textbox_font, border_width=2, wrap = "word")
        self.message_box.grid(row=5, column=1, padx=20, pady=(0, 10), sticky="nsew")

        ####################################################################
        # Image frame for classification result and image display
        self.image_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.image_frame.grid(row=1, column=2, rowspan=5, sticky="nswe")
        self.image_frame.grid_rowconfigure(1, weight=1)
        self.image_frame.grid_rowconfigure((2,3,4,5), weight=0)

        # Title for table
        self.title_label = customtkinter.CTkLabel(self, text="Classification Results", font=self.title_font)
        self.title_label.grid(row=0, column=2, padx=20, pady=(20, 10), sticky="nw")

        #Table frame
        self.result_table = customtkinter.CTkFrame(self.image_frame,border_width=2)
        self.result_table.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")
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

        # Show image
        self.show_single_image()
        self.add_message("Displaying example image: {}".format(os.path.split(self.image_path)[1]))
        # Show image path
        self.title_label = customtkinter.CTkLabel(self.image_frame, text= "Image: " + self.image_path, font=customtkinter.CTkFont(size=15))
        self.title_label.grid(row=6, column=0, padx=20, pady=(20, 10), sticky="nw")
        
    ####################################################################
    # Functions

    # Function to change appearance mode
    def change_appearance_mode_event(self):
        self.appearance_mode = self.appearance_mode_optionemenu.get()   
    
    # Function to get image path
    def get_image_path(self):
        self.image_path = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select an image file",  
                                                filetypes=(("jpg files", "*.jpg"),("png files", "*.png")))
        if self.image_path == "":
            self.add_message("Warning: Image is not selected, please select an image first!")
        self.show_single_image()
        

    # Function to choose and show single image
    def show_single_image(self):
        if self.image_path != "":
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
            self.add_message("Displaying selected image: {} ".format(os.path.split(self.image_path)[1]))
            self.single_classification()
        else:   
            # Display example image if not already selected
            self.image_path = os.path.expanduser(os.path.join(os.getcwd(), "example images","12102021111421-0012.jpg"))
            # self.add_message("Displaying example image: {}".format(os.path.split(self.image_path)[1]))
            self.single_image = Image.open(self.image_path)
        
        self.single_image = customtkinter.CTkImage(light_image=self.single_image,
                dark_image=self.single_image,
                size=(800, 800))
        image_label = customtkinter.CTkLabel(self.image_frame, image=self.single_image, text=None)  # display image with a CTkLabel
        image_label.grid(row=1, column=0, padx=20, pady=10, sticky = "nswe", rowspan=3)

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


    def checkbox_event(self):
        if self.checkbox_1.get():
            self.add_message("Crop Image Checked")
        else:
            self.add_message("Crop Image Unchecked")
    
    # Function to get image folder path
    def get_image_folder_path(self):
        if self.is_running:
            self.add_message("Warning: Batch classification is running!")
        else:
            self.image_folder_path = filedialog.askdirectory(initialdir=os.getcwd())
            if self.image_folder_path == "":
                self.add_message("Warning: Image folder is not selected, please select an image folder first!")
            else:
                self.add_message("Selected image folder path: {}".format(self.image_folder_path))

    # Function to save classification results
    def save_results(self):
        # creating a list of column names 
        # column_values = ['Filename','Output Data', 'Output Data','Confidence Score', 'Confidence Score','Weighted Score']
        if self.is_running:
            self.add_message("Warning: Batch classification is running!")
        elif self.class_results.size == 0:
            self.add_message("No results to save!")
        elif np.array_equal(self.class_results, self.class_results_prev):
            self.add_message("No changes to save!")
        else:
            self.class_results_prev = self.class_results.copy()
            column1 = ['', 'Output Data', 'Output Data','Confidence Score', 'Confidence Score', '']
            column2 = ['Filename', 'FBA', 'AVE', 'FBA', 'AVE', 'Weighted Score']
            column_values = pd.MultiIndex.from_arrays([column1,column2])

            # column_values = ['Filename','Output Data', 'Output Data','Output Data','Output Data','Output Data',
            #                  'Confidence Score', 'Confidence Score','Confidence Score','Confidence Score','Confidence Score','Weighted Score']

            # creating the dataframe 
            df = pd.DataFrame(data = self.class_results,  columns = column_values) 
            # print(df) 

            # datetime object containing current date and time
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
            save_path = os.path.join(os.getcwd(), 'results', dt_string+".csv")
            # print(save_path) 
            df.to_csv(save_path)
            self.add_message("Classification results saved to " + save_path)

    # Function to save classification results automatically
    def auto_save(self):

        if not self.is_running:
            column1 = ['', 'Output Data', 'Output Data','Confidence Score', 'Confidence Score', '']
            column2 = ['Filename', 'FBA', 'AVE', 'FBA', 'AVE', 'Weighted Score']
            column_values = pd.MultiIndex.from_arrays([column1,column2])

            # column_values = ['Filename','Output Data', 'Output Data','Output Data','Output Data','Output Data',
            #                  'Confidence Score', 'Confidence Score','Confidence Score','Confidence Score','Confidence Score','Weighted Score']
            # creating the dataframe 
            df = pd.DataFrame(data = self.class_results,  columns = column_values) 
            # print(df) 
            df.to_csv('autosave_results.csv')
            self.add_message("Classification results saved to autosave_results.csv")

    # Function to save classification results in the specified directory
    def save_as(self):
        if self.is_running:
            self.add_message("Warning: Batch classification is running!")
        elif self.class_results.size == 0:
            self.add_message("No results to save!")
        else:
            save_path = filedialog.asksaveasfilename(initialdir=os.getcwd(), defaultextension = ".csv", filetypes=(("csv files", "*.csv"),("all files", "*.*")))
            if save_path == "":
                self.add_message("Warning: File path is not selected, please select a file path first!")
            else:
                column1 = ['', 'Output Data', 'Output Data','Confidence Score', 'Confidence Score', '']
                column2 = ['Filename', 'FBA', 'AVE', 'FBA', 'AVE', 'Weighted Score']
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
        elif self.image_path == "":
            self.add_message("Warning: Image is not selected, please select an image first!")
        else:
            self.add_message("Single classification started...")

            # Set the flag to true when the classification is running
            self.is_running = True


            # Get the file path of the model
            model_name = "model.tflite"
            model_path = os.path.join(os.getcwd(), 'model', model_name)

            # Load the TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path)
            interpreter.allocate_tensors()

            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            img = cv2.imread(self.image_path)
            height, width, *_ = img.shape
            if self.checkbox_1.get():
                img = img[math.floor(height*0.2):math.floor(height*0.8), 0:width]
            new_img = cv2.resize(img, (224, 224))
            # input_details[0]['index'] = the index which accepts the input
            interpreter.set_tensor(input_details[0]['index'], [new_img])
            # run the inference
            interpreter.invoke()
            # output_details[0]['index'] = the index which provides the input
            output_data = interpreter.get_tensor(output_details[0]['index'])
            # Squeeze the output data to a 1D array
            output_data = np.squeeze(output_data)
            # calculate the confidence score using the output data
            confidence_score = output_data/256*100
            # calculate the weighted score using the confidence score
            weighted_score = np.dot(WEIGHTS, confidence_score/100)
            self.add_message("Image processed: {}, Output Data: {} Confinence Score: {}, Weighted Score: {}".
                             format(Path(self.image_path).stem, output_data, confidence_score.round(2), weighted_score.round(2)))

            # Update weighted score
            self.cell_weighted_score.configure(text=str((weighted_score).round(2)))

            # Update confidence score

            self.confidence_table[0].configure(text=str(confidence_score[0].round(2)))
            self.confidence_table[2].configure(text=str(confidence_score[1].round(2)))
            # For full model
            # for i in range(len(confidence_score)):
            #     self.confidence_table[i].configure(text=str(confidence_score[i].round(2)))


            # creating the Numpy array 
            results = []
            results = np.append(results,Path(self.image_path).stem)
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
        elif self.image_folder_path == "":
            self.add_message("Warning: Image folder is not selected, please select an image folder first!")        
        else:
            self.add_message("Batch classification started...")

            # Set the flag to true when the classification is running
            self.is_running = True

            # Get the file path of the model
            model_name = "model.tflite"
            model_path = os.path.join(os.getcwd(), 'model', model_name)

            # Load the TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path)
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

            for file in pathlib.Path(self.image_folder_path).iterdir():         
                # read and resize the image
                img = cv2.imread(r"{}".format(file.resolve()))
                height, width, *_ = img.shape
                if self.checkbox_1.get():
                    img = img[math.floor(height*0.2):math.floor(height*0.8), 0:width]
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

                confidence_score = output_data/256*100
                weighted_score = np.dot(WEIGHTS, confidence_score/100)

                self.add_message("Image processed: {}, Output Data: {} Confinence Score: {}, Weighted Score: {}".
                             format(os.path.split(self.image_path)[1], output_data, confidence_score.round(2), weighted_score.round(2)))

                np1 = np.squeeze(confidence_score)
                np2 = np.squeeze(weighted_score)
                filenames = np.append(filenames, file.stem)

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
            self.auto_save()


if __name__ == "__main__":
    app = App()
    app.mainloop()
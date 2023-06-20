
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from mpl_interactions import ioff, panhandler, zoom_factory
import numpy as np
import pandas as pd
from math import radians, sin, cos, acos
import ast

class Animation:
    def __init__(self):

        ########### กำหนดการแสดงผลของกราฟ ###########################
        fig, ax = plt.subplots() # สร้าง figure และ axes โดย fig เป็นพื้นที่ที่แสดงกราฟทั้งหมด, ax เป็นพื้นที่ที่ใช้ในการแสดงเนื้อหากราฟ เช่น เส้นกราฟ แกน x และแกน y 
        fig.subplots_adjust(bottom=0.2) # ปรับแต่งตำแหน่งของ subplot ให้มีพื้นที่ว่างด้านล่างของกราฟ 0.2

        ax.set_title('GPS and Lidar playback') # กำหนดชื่อกราฟ

        plt.xlim(-10, 10) # กำหนดช่วงแกน x ให้อยู่ในช่วง -10 ถึง 10 
        plt.ylim(-10, 10) # กำหนดช่วงแกน y ให้อยู่ในช่วง -10 ถึง 10
        plt.grid() # แสดงตาราง
        plt.axis('square') # กำหนดให้สเกลแกน x และ y เท่ากัน
        plt.xlabel("Distance [m]") # กำหนดชื่อแกน 
        plt.ylabel("Distance [m]") # กำหนดชื่อแกน

        self.disconnect_zoom = zoom_factory(ax) # เรียกใช้ฟังชันการ zoom ให้สามารถ zoom ได้ด้วยการ scrolling

        self.pan_handler = panhandler(fig) # เรียกใช้ฟังชันการ panning ให้สามารถเลื่อนเพื่อดูกราฟได้ด้วยการคลิกขวาค้าง

        self.animation = animation.FuncAnimation( # สร้าง animation โดยใช้ FuncAnimation
            fig, self.update,frames = None, interval=1000) # โดยมีจำนวนเฟรม = inf และอัพเดท animation ทุก ๆ 1000ms = 1s
        

        ########### สร้างจุดหรือเส้นบนกราฟ ###########################
        self.line_x, = ax.plot([], [], lw=1, color='green', label ='Heading') # สร้างเส้นกราฟสี เขียว ที่แสดงถึง heading โดยยังไม่กำหนดจุดเริ่มต้นและจุดปลายของเส้น 

        self.point, = ax.plot([], [], 'ro', markersize=2,  label ='Lidar') # สร้างวงกลมสี แดง ขนาด 2 บนกราฟเพื่อแสดงถึงข้อมูล Lidar

        self.point_robot, = ax.plot([], [], 'bo', markersize=9, label ='Robot')  # สร้างวงกลมสี น้ำเงืน ขนาด 9 บนกราฟเพื่อแสดงถึงข้อมูลตำแหน่งของหุ่นยนต์

        self.point_path, = ax.plot([], [], 'o',color='orange', markersize=2, label ='Path')  # สร้างวงกลมสี ส้ม ขนาด 2 บนกราฟเพื่อแสดงถึงข้อมูล path การเดินของหุ่นยนต์
        
        ax.legend(loc ="upper left") # วางคำอธิบายข้อมูลกราฟไว้ที่มุมุขวาบน


        ########### สร้างสัญลักษณ์เข็มทิศ ###########################
        ax.text(0.5, 1.5, 'N', fontsize=16,ha='center', va='center') # สร้างสัญลักษณ์ N เพื่อแสดงถึงทิศเหนือ
        
        x0, y0 = 0, 0 # กำหนดพิกัด x, y ของจุดเริ่มต้นของลูกศร
        
        x1, y1 = 0, 1 # กำหนดพิกัด x, y ของจุดสิ้นสุดของลูกศร
        
        arrow = ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.5, head_length=0.5, fc='black', ec='black') # สร้างลูกศร
        
        
        ############### อ่านข้อมูล GPS, Lidar จากไฟล์ .csv และ คำนวณการเคลื่อนที่ของหุ่นยนต์ และแสดงข้อมูลของ lidar #########################################
            
        self.lidar_data = pd.read_csv('ydlidar_20230612164330.csv') # อ่านข้อมูล Lidar

        self.gps_data = pd.read_csv('gpsPlus_20230612164330.csv') # อ่านข้อมูล GPS

        self.slat = radians(float(self.gps_data['gps_recentLatitudeN'][0])) # ดึงค่าแรกของข้อมูล Latitude 

        self.slon = radians(float(self.gps_data['gps_recentLongitudeE'][0])) # ดึงค่าแรกของข้อมูล Longitude


        ################################ initial ตำแหน่งและทิศทางการหมุนของหุ่นยนต์ ###########################

        self.transform_mtx = np.array([[1, 0, 0, 0],    # สร้าง Homogenous Transformation Matrix
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]])
        


        ################################ initial ทิศทางการหมุนของหุ่นยนต์อีกครั้ง ###########################
        self.heading = np.deg2rad(90) # ให้ทิศเหนือ (N) เป็นทิศเริ่มต้นของหุ่นยนต์ โดยการหมุน 90 องศาเพื่อให้ทิศทางการหมุนตั้งฉากกับแกน x

        self.transform_mtx = np.dot(self.transform_mtx,np.array([[cos(self.heading), -sin(self.heading), 0, 0], # นำ matrix มา dot กันเพื่อหาทิศทางปัจจุบัน
                                                                 [sin(self.heading),  cos(self.heading), 0, 0], # โดยหมุนแกนในแนวแกน Z
                                                                 [0,             0,            1, 0],
                                                                 [0,             0,            0, 1]]))
        
        self.heading = np.deg2rad(-1*self.gps_data['compass_heading_degs'][0])  # หมุนด้วยทิศทางแรกของข้อมูล heading ด้วยข้อมูล compass

        self.transform_mtx = np.dot(self.transform_mtx,np.array([[cos(self.heading), -sin(self.heading), 0, 0], # นำ matrix มา dot กันเพื่อหาทิศทางปัจจุบัน
                                                                 [sin(self.heading),  cos(self.heading), 0, 0], # โดยหมุนในแนวแกน Z
                                                                 [0,             0,            1, 0],
                                                                 [0,             0,            0, 1]]))




        ########################## คำนวณข้อมูลการเคลื่อที่ของหุ่นยนต์ตั้งแต่เริ่มต้นจนจบ จากไฟล์ .csv ##########################
        self.distance = 0.0
        self.lastdistance = 0.0

        self.backup_x = []
        self.backup_y = []

        self.backup_x_position = []
        self.backup_y_position = []

        self.backup_x_lidar = []
        self.backup_y_lidar = []
        
        for count in range(1,len(self.gps_data)):
            elat = radians(float(self.gps_data['gps_recentLatitudeN'][count])) # อ่านค่า Latitude ปัจจุบัน
            elon = radians(float(self.gps_data['gps_recentLongitudeE'][count])) # อ่านค่า Longitude ปัจจุบัน

            heading = np.deg2rad(-1*(self.gps_data['compass_heading_degs'][count] - self.gps_data['compass_heading_degs'][count-1])) # คำนวณหาทิศทางที่เปลี่ยนไป แล้วคูณด้วย -1 เพื่อเปลี่ยนจาก left-handed coordinate เป็น right-handed แนนพกรืฟะำ

            self.distance = (6371.01 * acos(sin(self.slat)*sin(elat) + cos(self.slat)*cos(elat)*cos(self.slon - elon))) # คำนวณหาระยะทางจากจุดเริ่มต้นจากข้อมูล GPS

            dist = abs(self.distance*1000 - self.lastdistance*1000) # หาระยะทางที่เปลี่ยนไป
         
            self.lastdistance = self.distance # เก็บค่าระยะทางล่าสุด

            translation_mtx = np.array([[1, 0, 0, dist],  # เตรียม Homogenous Transformation Matrix สำหรับการ translation
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        
            rotation_mtx = np.array([[cos(heading), -sin(heading), 0, 0],   # เตรียม Homogenous Transformation Matrix สำหรับการ rotation
                                    [sin(heading),  cos(heading), 0, 0],
                                    [0,             0,            1, 0],
                                    [0,             0,            0, 1]])
            
            ########## dot matrix เพื่อหาตำแหน่งและทิศทางปัจจุบัน ################
            self.transform_mtx = np.dot(self.transform_mtx,translation_mtx)
            self.transform_mtx = np.dot(self.transform_mtx,rotation_mtx)


            ########### คำนวณจุดต้นและจุดปลายของเส้นที่แสดง heading ของหุ่นยนต์ (เส้นสีเขียว) ##################################
            angle = np.arctan2(self.transform_mtx[1][0],self.transform_mtx[1][1]) # คำนวณหา heading ของหุ่นยนต์

            x = [self.transform_mtx[0][3],(self.transform_mtx[0][3]+(10*np.cos(angle)))] # [x_start_point,x_end_point]
            y = [self.transform_mtx[1][3],(self.transform_mtx[1][3]+(10*np.sin(angle)))] # [y_start_point,y_end_point]

            ############################ LIDAR ############################# 
            distances = self.lidar_data['lidar_range_meter'][count]  # อ่านค่า range ของแต่ละ angle ในปัจจุบัน
            distances = ast.literal_eval(distances) #convert the string to array
            distances = np.array(distances)


            angles = self.lidar_data['lidar_angle_degree'][count] # อ่านค่า angle ปัจจุบัน
            angles = ast.literal_eval(angles) #convert the string to array
            angles = np.array(angles)
            angles = np.deg2rad(angles)

            # แปลงข้อมูล Lidar ที่เป็น Polar coordinate เป็น cartesian coordinate
            x_lidar = self.transform_mtx[0][3] + (distances * np.cos((-angles + angle))) # x_lidar = range_lidar * (มุม Lidar + มุม heading ที่เปลี่ยนไป) 
            y_lidar = self.transform_mtx[1][3] + (distances * np.sin((-angles + angle))) # y_lidar = range_lidar * (มุม Lidar + มุม heading ที่เปลี่ยนไป)

            ########### เก็บข้อมูลไว้ใน list ทุกครั้ง ##########################################
            self.backup_x.append(x)
            self.backup_y.append(y)

            self.backup_x_position.append(self.transform_mtx[0][3]) # การเคลื่อนที่แนวแกน X 
            self.backup_y_position.append(self.transform_mtx[1][3]) # การเคลื่อนที่แนวแกน Y 

            self.backup_x_lidar.append(x_lidar)
            self.backup_y_lidar.append(y_lidar)
        


        self.count = 0 # ใช้สำหรับการนับลำดับ
       
        ############# slider #########################################
        self.slideraxis = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(self.slideraxis, label='Time [sec]',
                        valmin=0, valmax=len(self.gps_data)-2, valinit=0)
        self.slider.on_changed(self.onChange)


        ################## buttons ################################
        self.axes = plt.axes([0.1, 0.5, 0.1, 0.075])
        self.bnext = Button(self.axes, 'Robot path',color="orange")
        self.bnext.on_clicked(self.add)

        self.axes_lidar = plt.axes([0.1, 0.8, 0.1, 0.075])
        self.bnext_lidar = Button(self.axes_lidar, 'Lidar',color="red")
        self.bnext_lidar.on_clicked(self.show_lidar_button)

        self.axes_play_pause = plt.axes([0.1, 0.1, 0.075, 0.075])
        self.bnext_play_pause = Button(self.axes_play_pause, 'Play/Pause')
        self.bnext_play_pause.on_clicked(self.play_pause_button)

        ############## initial สถานะการแสดงข้อมูลกราฟ ########
        self.show_path = True
        self.show_lidra = True
        self.play_pause = True


    def onChange(self, value):
        self.count = int(value) # รับค่าจาก Slider
  
    def add(self, val):
        self.show_path = not self.show_path # show/don't show path เมื่อกดปุ่ม Robot path
    
    def show_lidar_button(self, val):
        self.show_lidra = not self.show_lidra # show/don't show ข้อมูล Lidar เมื่อกดปุ่ม Lidar
    
    def play_pause_button(self, val):   # หยุด/เล่น animation เมื่อกดปุ่ม Play/Pause
        self.play_pause = not self.play_pause

        if  self.play_pause:
            self.animation.resume()
        else:
            self.animation.pause()
        
    def update(self, i):

        if self.count < len(self.gps_data)-2:
            self.count = self.count + 1    # นับเวลาครั้งละ 1 วินาที

        self.slider.set_val(self.count) # อัพเดทค่า slider ตามเวลาปัจจุบัน
        
         # แสดง/ไม่แสดง วงกลมสี ส้ม บนกราฟเพื่อแสดงถึงข้อมูล path การเดินของหุ่นยนต์
        if self.show_path:
            self.point_path.set_data(self.backup_x_position[0:self.count], self.backup_y_position[0:self.count])
        else:
            self.point_path.set_data(None, None)
        
        # แสดงเส้น heading
        self.line_x.set_data(self.backup_x[self.count], self.backup_y[self.count])
        
        # แสดง/ไม่แสดง วงกลมสี แดง บนกราฟเพื่อแสดงถึงข้อมูล Lidar
        if self.show_lidra:
            self.point.set_data(self.backup_x_lidar[self.count], self.backup_y_lidar[self.count])
        else:
            self.point.set_data(None, None)

        # แสดงตำแหน่งของหุ่นยนต์
        self.point_robot.set_data(self.backup_x_position[self.count], self.backup_y_position[self.count])


pa = Animation()
plt.show()
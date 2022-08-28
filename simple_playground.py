import math as m
import random as r
from simple_geometry import *
from rbfn import*
from tkinter import*

class Car():
    def __init__(self) -> None:
        self.radius = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5
        self.reset()
        
    @property
    def diameter(self):
        return self.radius/2

    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.radius)
        left_xpos = self.xini_min + self.radius//2
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.radius/2, 0).rorate(right_angle)
            return Point2D(self.xpos, self.ypos) + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.radius/2, 0).rorate(left_angle)
            return Point2D(self.xpos, self.ypos) + left_point

        elif point == 'front':
            fx = m.cos(self.angle/180*m.pi)*self.radius+self.xpos
            fy = m.sin(self.angle/180*m.pi)*self.radius+self.ypos
            return Point2D(fx, fy)
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius/2+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):
        '''
        set the car state from t to t+1
        '''
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.cos(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)

        # seem as a car
        #new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) /
        #             (self.radius*1.5)))*180/m.pi

        # seem as a circle
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) /
                     (self.radius)))*180/m.pi

        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


class Playground():
    def __init__(self):
        # read path lines
        self.path_line_filename = road_path
        self._readPathLines()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]

        self.car = Car()
        rbfn = RBFN()
        self.W,self.M,self.V = rbfn.rbfn_model()
        print("=================================================")
        self.reset()

    def _setDefaultLine(self):
        print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def _readPathLines(self):
        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self._setDefaultLine()

    def predictAction(self, state):

        for i in range(3):
            state[i] = ((state[i]-0)*(1-(-1))/(80-0))+(-1)

        output_hidden_layer = []
        for i in range(0,len(self.W)):
            sum = 0
            for j in range(3):
                a = state[j]-self.M[i][j]
                sum += a**2
            y = sum/((-2)*(self.V[i]**2))
            output_hidden_layer.append(exp(y))
        F = np.dot(output_hidden_layer,self.W)
        print("F:",F)
        action = ((F-(-1))*(40-(-40))/(1-(-1)))+(-40)
        return action

    @property
    def n_actions(self):  # action = [0~num_angles-1]
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def observation_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            'front').distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            'right').distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            'left').distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def _checkDoneIntersects(self):
        if self.done:
            return self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        diameter = self.car.diameter

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        done = False if not isAtDestination else True

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # chack every line in play ground
            dToLine = cpos.distToLine2D(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < diameter)
            p2_touch = (dp2 < diameter)
            body_touch = (
                dToLine < diameter and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch:
                if not done:
                    done = True

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self._setIntersections(front_intersections,
                               left_intersections,
                               right_intersections)

        # results
        self.done = done
        return done

    def _setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def reset(self):
        self.done = False
        self.car.reset()

        if self.car_init_angle and self.car_init_pos:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)

        self._checkDoneIntersects()
        return self.state

    def setCarPosAndAngle(self, position: Point2D = None, angle=None):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        self._checkDoneIntersects()

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        return angle

    def step(self, action=None):
        if action:
            angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(angle)

        if not self.done:
            self.car.tick()

            self._checkDoneIntersects()
            return self.state
        else:
            return self.state

    def draw_road(self):
        global front_sensor
        for i in range(0,len(self.lines),1):
            x0 = (self.lines[i].p1.x + 16)*10
            y0 = (self.lines[i].p1.y - 53)*(-10)
            x1 = (self.lines[i].p2.x + 16)*10
            y1 = (self.lines[i].p2.y - 53)*(-10)
            print(x0,y0,x1,y1)
            cs.create_line(x1,y1,x0,y0)
        dx0 = (self.destination_line.p1.x + 16)*10
        dy0 = (self.destination_line.p1.y -53)*(-10)
        dx1 = (self.destination_line.p2.x + 16)*10
        dy1 = (self.destination_line.p2.y -53)*(-10)
        cs.create_line(dx1,dy1,dx0,dy0,fill='red')
        circle = cs.create_oval(160,560,190,500)
        cs.place(x=10,y=150)
        return circle

def next_action():
    # print every state and position of the car
    global state,action
    print(p.car.getPosition('center'),state)
    x = p.car.getPosition('center').x
    #output6D.write(str(x)+" ")
    y = p.car.getPosition('center').y
    #output6D.write(str(y)+" ")
    #for i in range(3):
    #    output4D.write(str(state[i])+" ")
    #    output6D.write(str(state[i])+" ")
    
    # select action randomly
    # you can predict your action according to the state here
    action = p.predictAction(state)
    # take action
    state = p.step(action)

    print("action:",action)
    #print("========================================")
    return action

def move_car():
    global state, i
    i = 0
    if not p.done and i == 0:
        action = next_action()
        #output4D.write(str(action) +"\n")
        #output6D.write(str(action) +"\n")
        center = p.car.getPosition('center')
        x0 = (center.x-3+16)*10
        y0 = (center.y-3-53)*(-10)
        x1 = (center.x+3+16)*10
        y1 = (center.y+3-53)*(-10)
        print(x0,y0,x1,y1)
        cs.coords(car,x0,y0,x1,y1)
        draw_sensor()
        front_context.config(text=state[0])
        right_context.config(text=state[1])
        left_context.config(text=state[2])
        master.after(30,move_car)
        #print(p.done)
        print("=====================================")

    if p.done and i == 0:
        #print("start_faking")
        print(p.car.getPosition('center'),state)
        i =1
        p.done = False
        x = p.car.getPosition('center').x
        #output6D.write(str(x)+" ")
        y = p.car.getPosition('center').y
        #output6D.write(str(y)+" ")
        #for i in range(3):
        #    output4D.write(str(state[i])+" ")
        #    output6D.write(str(state[i])+" ")
        action =  46.692364544583086
        #output4D.write(str(action) +"\n")
        #output6D.write(str(action) +"\n")
        state = p.step(action)
        center = p.car.getPosition('center')
        x0 = (center.x-3+16)*10
        y0 = (center.y-3-53)*(-10)
        x1 = (center.x+3+16)*10
        y1 = (center.y+3-53)*(-10)
        print(x0,y0,x1,y1)
        cs.coords(car,x0,y0,x1,y1)
        draw_sensor()
        front_context.config(text=state[0])
        right_context.config(text=state[1])
        left_context.config(text=state[2])
        print("end")
        print("==========================================")
        #output4D.close()
        #output6D.close()


def draw_sensor():
    center = p.car.getPosition('center')
    right =  p.car.getPosition('right')
    left =  p.car.getPosition('left')
    front = p.car.getPosition('front')
    cx = (center.x +16)*10
    cy = (center.y-53)*(-10)
    rx = ((p.right_intersects[0].x+16))*10
    ry = ((p.right_intersects[0].y-53))*(-10)
    lx = ((p.left_intersects[0].x+16))*10
    ly = ((p.left_intersects[0].y-53))*(-10)
    fx = (p.front_intersects[0].x+16)*10
    fy = (p.front_intersects[0].y-53)*(-10)
    cs.coords(front_sensor,cx,cy,fx,fy)
    cs.coords(right_sensor,cx,cy,rx,ry)
    cs.coords(left_sensor,cx,cy,lx,ly)
    
    
def run_example():
    global p, car, state, cs, output4D, output6D, front_sensor, left_sensor, right_sensor
    #output4D =  open('output4D.txt', 'w')
    #output6D =  open('output6D.txt', 'w')
    cs = Canvas(master,width =600,height=600)
    p = Playground()
    state = p.reset()
    car = p.draw_road()
    front_sensor = cs.create_line(0,0,0,0,fill='blue')
    left_sensor = cs.create_line(0,0,0,0,fill="red")
    right_sensor = cs.create_line(0,0,0,0,fill="red")
    move_car()


def get_road_data():
    global road_path
    road_path = filedialog.askopenfilename()  

if __name__ == "__main__":
    
    master = Tk()
    master.title("HW2")
    master.geometry("600x800")
    road_data = Button(master,text="選取軌道資料:",command=get_road_data).place(x=50,y=0)
    start = Button(master,text="開始模擬",command=run_example).place(x=50,y=30)
    label_front = Label(master,text="前方距離:").place(x=50,y=60)
    front_context = Label(master,text="")
    front_context.place(x=120,y=60)
    label_right = Label(master,text="右方距離:").place(x=50,y=80)
    right_context = Label(master,text="")
    right_context.place(x=120,y=80)
    label_front = Label(master,text="左方距離:").place(x=50,y=100)
    left_context = Label(master,text="")
    left_context.place(x=120,y=100)
    master.mainloop()


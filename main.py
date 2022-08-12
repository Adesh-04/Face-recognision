import cv2
from random import randrange

def main():

    trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # img = cv2.imread('image1.jpeg')
    # print(img)

    webcam = cv2.VideoCapture(0)

    while True:
        succesful_frame_read, current_frame = webcam.read()



        # greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        greyscaled_img = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        face_coordinate = trained_face_data.detectMultiScale(greyscaled_img)

        print(face_coordinate)

        for coord in face_coordinate:
            (x,y,h,w) = coord
            cv2.rectangle(current_frame, (x, y), (x+h, y+w), (randrange(255), randrange(255), randrange(255)), 2)

        # rectangle( [image], [top_left_corner_coordinate], [bottom_right_corner_coordinate + top_left_corner_coordinate], [colour(BGR)], [thickness] )
        # face_coordinate = [[ 75 96 327 327]]
        # cv2.rectangle(img, (75 , 96), (75+327, 96+327), (0, 255, 0), 2)


        cv2.imshow('Clever Programmer Face Detector', current_frame)

        key = cv2.waitKey(1)

        # q or Q as a ASCII
        if key == 81 or key == 113:
            break


    


    


if __name__ == '__main__':
    main()

    print("Code Completed")
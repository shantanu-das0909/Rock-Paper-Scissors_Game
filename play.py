import cv2
import random
import numpy as np
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture(0)

model = load_model("rock-paper-scissors-model.h5")

CLASS_MAPPER = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return CLASS_MAPPER[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "You"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "You"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "You"
        if move2 == "rock":
            return "Computer"


def prediction(move):
    pred = model.predict(move)
    move_code = np.argmax(pred[0])
    show_string = mapper(move_code)
    return show_string


def computer_move():
    n = random.randint(0, 2)
    comp_move = mapper(n)
    return comp_move


prev_move = None
while True:
    success, frame = cap.read()
    frame = cv2.resize(frame, (1300, 500))
    cv2.rectangle(frame, (100, 100), (500, 500), (255, 255, 255), 2)
    cv2.rectangle(frame, (800, 100), (1200, 500), (255, 255, 255), 2)

    image = frame[100:500, 100:500]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = np.array([image])

    my_move = prediction(image)

    if prev_move != my_move:
        if my_move != "none":
            computer_move_name = computer_move()
            winner = calculate_winner(my_move, computer_move_name)
        else:
            computer_move_name = "none"
            winner = "Waiting..."
    prev_move = my_move

    user_input = "Your move : " + "  " + my_move.upper()
    comp_input = "Computer move : " + "  " + computer_move_name.upper()

    cv2.putText(frame, user_input, (150, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "WINNER", (580, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (14, 118, 250), 2, cv2.LINE_AA)
    cv2.putText(frame, winner.upper(), (580, 90), cv2.FONT_HERSHEY_COMPLEX, 1, (14, 118, 250), 2, cv2.LINE_AA)
    cv2.putText(frame, comp_input, (850, 50), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    if computer_move_name != "none":
        icon = cv2.imread(
            "images/{}.png".format(computer_move_name))
        icon = cv2.resize(icon, (400, 400))
        frame[100:500, 800:1200] = icon

    cv2.imshow("video", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

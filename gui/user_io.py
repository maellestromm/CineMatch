from gui.backend import get_recommendations_from_profile, print_recs
from letterboxdpy import user
import time


# User I/O in terminal
def main():
    start_time, recs = 0, []
    # loop until user inputs valid Letterboxd username
    y = True
    while y:
        # create user instance with username
        username = input("Enter Letterboxd username: ")

        try:
            # ensure user is real
            user.User(username)

            # get recommendations for user
            start_time = time.time()
            recs = get_recommendations_from_profile(username)
            y = False
        except:
            print("This user does not exist or their reviews are unavailable.")

    end_time = (time.time() - start_time)
    print_recs(recs)
    print(f"Response time: {end_time:>6.4f} s\n")


if __name__ == "__main__":
    main()

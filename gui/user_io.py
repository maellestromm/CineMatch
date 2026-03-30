from .backend import get_recommendations_from_profile, print_recs
from letterboxdpy import user

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    
    # loop until user inputs valid Letterboxd username
    y=True
    while(y==True):
        # create user instance with username
        username = input("Enter Letterboxd username: ")

        try:
            # ensure user is real
            user_instance = user.User(username)

            # get recommendations for user
            recs = get_recommendations_from_profile(username)
            y=False
        except:
            print("This user does not exist or their reviews are unavailable.")

    print_recs(recs)
    

if __name__ == "__main__":
    main()
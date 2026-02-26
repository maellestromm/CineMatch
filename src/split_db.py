import sqlite3
import pandas as pd
import shutil
import random
import os

# --- Configuration ---
ORIGINAL_DB = "../data/user_first_cut2_clear.db"
TRAIN_DB = "train_model.db"
TEST_DB = "test_eval.db"
TEST_RATIO = 0.2  # 20% of users go to the test database
MIN_REVIEWS = 10  # Only consider users with at least 10 reviews for testing


def split_database():
    print("🪓 [Split] Starting database physical split...")

    # 1. Copy the original DB to create Train and Test DBs
    print("[Split] Copying files...")
    shutil.copyfile(ORIGINAL_DB, TRAIN_DB)
    shutil.copyfile(ORIGINAL_DB, TEST_DB)

    # 2. Get all valid users from the original database
    conn = sqlite3.connect(ORIGINAL_DB)
    query = "SELECT user_username, COUNT(id) as review_count FROM reviews GROUP BY user_username HAVING review_count >= ?"
    df_users = pd.read_sql_query(query, conn, params=[MIN_REVIEWS,])
    conn.close()

    all_users = df_users['user_username'].tolist()

    # 3. Randomly select test users
    num_test_users = int(len(all_users) * TEST_RATIO)
    test_users = random.sample(all_users, num_test_users)
    train_users = list(set(all_users) - set(test_users))

    print(f"[Split] Total valid users: {len(all_users)}")
    print(f"[Split] Allocating {len(train_users)} users to Train DB.")
    print(f"[Split] Allocating {len(test_users)} users to Test DB.")

    # 4. Clean up TRAIN_DB (Delete all test users)
    print("🧹 [Split] Removing test users from Train DB...")
    conn_train = sqlite3.connect(TRAIN_DB)
    cursor_train = conn_train.cursor()
    placeholders = ','.join(['?'] * len(test_users))
    cursor_train.execute(f"DELETE FROM reviews WHERE user_username IN ({placeholders})", test_users)
    cursor_train.execute(f"DELETE FROM users WHERE username IN ({placeholders})", test_users)
    conn_train.commit()
    cursor_train.execute("VACUUM")
    conn_train.close()

    # 5. Clean up TEST_DB (Delete all train users)
    print("🧹 [Split] Removing train users from Test DB...")
    conn_test = sqlite3.connect(TEST_DB)
    cursor_test = conn_test.cursor()
    placeholders = ','.join(['?'] * len(train_users))
    cursor_test.execute(f"DELETE FROM reviews WHERE user_username IN ({placeholders})", train_users)
    cursor_test.execute(f"DELETE FROM users WHERE username IN ({placeholders})", train_users)
    conn_test.commit()
    cursor_test.execute("VACUUM")
    conn_test.close()

    print("✅ [Split] Database split successfully completed!")


if __name__ == "__main__":
    split_database()
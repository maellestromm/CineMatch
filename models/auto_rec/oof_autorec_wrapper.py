from util import root_path
from .train_autorec import train_model
from .infer_autorec import AutoRecRecommender


def get_oof_autorec(db_path, temp_weights_path=root_path() / "data/temp_oof_autorec.pth",
                    temp_json_path=root_path() / "data/temp_oof_autorec.json"):
    print(f"\n[Wrapper] re-training autorec: {db_path}")

    train_model(
        train_db=db_path,
        test_db=db_path,
        weights_save_path=temp_weights_path,
        dict_path=temp_json_path,
        epochs=113
        # The database was not split here. Based on the previous situation,
        # 113 epochs was the maximum number of rounds before overfitting, so we truncated it here.
    )

    print("[Wrapper] re-training autorec done")
    return AutoRecRecommender(db_path=db_path, weights_path=temp_weights_path, dict_path=temp_json_path)

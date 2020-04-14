import os
import pressure

if os.path.isfile("raw/label_list"):
    print("Error: label_list already exists.")

elif os.path.isfile("raw_feature_list"):
    print("Info: raw_feature_list found.")
    with open("raw/raw_feature_list", "r") as raw_features, open("raw/label_list", "a") as features:
        for line in raw_features:
            content = line.split()

            raw_pen_pressure = float(content[0])
            page_id = content[1]

            pen_pressure, comment = pressure.determine_pen_pressure(raw_pen_pressure)

            features.write("%s\t" % str(raw_pen_pressure))
            features.write("%s\t" % str(pen_pressure))
            features.write("%s\t" % str(page_id))
            print('', file=features)
    print("Done!")

else:
    print("Error: raw_feature_list file not found.")

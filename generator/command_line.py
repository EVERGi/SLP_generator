from slp_generator import Generator
import argparse


model_dir = 'models/'
scaler_dir = 'scalers/'
input_dir = 'data/'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Clustering Command Line Tool")
    parser.add_argument("--evening", type=float, required=True, help="Use of building in the evening (0.0 to 1.0)")
    parser.add_argument("--weekend", type=float, required=True, help="Use of building in the weekends (0.0 to 1.0)")
    parser.add_argument("--yearly-consumption", type=int, required=True, help="Yearly consumption in kWh")
    parser.add_argument("--building-type", type=str, required=True, help="Type of building")
    args = parser.parse_args()
    evening, weekend, new_cons, type_cons = args.evening, args.weekend, args.yearly_consumption, args.building_type
    gen = Generator(input_dir)

    if evening is None or weekend is None:
        evening = None
        weekend = None
    scaled_consumption = gen.scaler.transform([[new_cons]])[0][0]
    base_load = gen.get_profile(scaled_consumption, type_cons, evening, weekend)
    new_load = base_load * new_cons

    # Change name of load
    new_load = new_load.rename(
        columns={"Power (kW)": "Customer_electric"}
    )

    # Save results
    name = "/Consumer.csv"
    new_load.to_csv(name)

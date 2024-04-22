import { GeneralHeader } from "./GeneralHeader";
import { NumberInput, TextInput } from "@mantine/core";
export const GeneralSettingsComponent = () => {
    return (
        <div>
            <GeneralHeader />
            <div className="grid grid-cols-3 gap-6">
                <div>
                    <NumberInput
                        label="Grafana context time radius"
                        description="Controls how far back in time the Grafana context will go in minutes."
                        placeholder="2"
                        mt="xs"
                        onBlur={(event) => {
                            localStorage.setItem(
                                "grafanaContextTimeRadius",
                                event.target.value
                            );
                        }}
                    />
                </div>
            </div>
        </div>
    );
};

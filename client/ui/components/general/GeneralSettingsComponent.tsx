import { GeneralHeader } from "./GeneralHeader"
import { NumberInput, TextInput } from "@mantine/core"
export const GeneralSettingsComponent = () =>  {
    return (
        <div>
            <GeneralHeader/>
            <div className="grid grid-cols-3 gap-6">
                <div>
                    <TextInput
                        label="Grafana URL"
                        fz="sm"
                        mt="xs"
                    />
                    <TextInput
                        label="Loki ID"
                        fz="sm"
                        mt="xs"
                    />
                    <NumberInput
                        label="Context Size"
                        placeholder="2"
                        mt="xs"
                    />
                </div>
            </div>
        </div>
        
        
    )
}
import { trpc } from "../../../utils/trpc";
import { ServiceCard } from "./ServiceCard";
import { useState } from "react";
import { ExtendedView } from "./ExtendedView";
import { ListHeader } from "./ListHeader";
import { Service } from "@/components/server/models/service";

export const ListServices = () => {
    const { data: services } = trpc.services.getServices.useQuery(undefined, {
        staleTime: 5 * 1000,
        select: (data) => data?.data,
    });

    const [selectedService, setSelectedService] = useState<Service | null>(
        null
    );

    const handleConfigClick = (service: Service | null) => {
        setSelectedService(service);
    };

    const handleGoBackToList = () => {
        setSelectedService(null);
    };

    return (
        <div>
            {selectedService ? (
                <ExtendedView
                    service={selectedService}
                    onGoBack={handleGoBackToList}
                />
            ) : (
                <div>
                    <ListHeader />
                    <div className="grid grid-cols-3 gap-6">
                        {services?.map((service) => (
                            <ServiceCard
                                key={service._id}
                                service={service}
                                onConfigClick={handleConfigClick}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

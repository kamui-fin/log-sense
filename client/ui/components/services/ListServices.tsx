import { trpc } from "../../../utils/trpc";
import { ServiceCard } from "./ServiceCard";
import { useState } from "react";
import { ExtendedView } from "./ExtendedView";
import { ListHeader } from "./ListHeader";

export const ListServices = () => {
    const { data: services } = trpc.config.getServices.useQuery("getServices", {
        staleTime: 5 * 1000,
        select: (data) => data?.data,
    });

    const [selectedService, setSelectedService] = useState(null);

    const handleConfigClick = (service) => {
        console.log(service)
        setSelectedService(service);
    };

    const handleGoBackToList = () => {
        setSelectedService(null);
    };

    return (
        <div>
            {selectedService ? (
                <ExtendedView service={selectedService} onGoBack={handleGoBackToList} />
            ) : (
                <div>
                    <ListHeader />
                    <div className="grid grid-cols-3 gap-6">
                        {services?.map((service) => (
                            <ServiceCard
                                key={service.id}
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

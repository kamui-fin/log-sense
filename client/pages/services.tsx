import { Layout } from "../components/Layout";
import { NavbarSimple } from "../components/NavbarSimple";
import { ServiceCard } from "../components/ServiceCard";

const services = [
  {
    name: "HDFS",
    description: "Hadoop Distributed File System",
    mode: "train",
    threshold: -470.8,
    coresetNumber: 2,
  },
  {
    name: "Linux",
    description: "Linux logs from various nodes in AWS cloud",
    mode: "test",
    threshold: -340.2,
    coresetNumber: 10,
  },
];

const ServiceList = () => {
  return (<div className="grid grid-cols-3 gap-6">
    {
        services.map(service => {
            return (
                <ServiceCard service={service} />
            );
        })
    }

  </div>);
};

const ManageServices = () => {
  return (
    <div>
      <h1>Manage Services</h1>
      <ServiceList />
    </div>
  );
};

export default ManageServices;

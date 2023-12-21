import VHeader from '../Volunteer_Header/Volunteer_Header.jsx';
import './Volunteer_UI.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faClipboardCheck, faClipboardList } from '@fortawesome/free-solid-svg-icons';
import Timeline from './Timeline.jsx';
import Active_Timeline from './Active_Timeline.jsx';

export default function Volunteer_UI(){

    return(
        <body>
            <VHeader num={4}/>
            <section className="push"></section>
        

            <div className="row volunteer-cards"> {/* CARDS */}
                {/* SECOND CARD */}
                <div className="col card create-task mt-3 mx-4">
                    <FontAwesomeIcon icon={faClipboardList} size="2xl" />
                    <p></p>
                    
                    <h5 className="card-title">AVAILABLE TRIPS</h5>
                    <div className="card-body">
                        <Timeline></Timeline>
                    </div>
                </div>
                <div className="col card create-task mt-3 mx-4">
                <FontAwesomeIcon icon={faClipboardCheck} size="2xl" />
                <p></p>
                <h5 className="card-title">ACTIVE TRIPS</h5>
                    <div className="card-body">
                        <Active_Timeline></Active_Timeline>
                    </div>
                </div>
            </div>
            <div className="row"> {/* MAP */}
                <div className="col section-header my-2 mx-0">
                <div className="container-fluid map-container-2">
                    <iframe src="https://maps.google.com/maps?q=chicago&t=&z=13&ie=UTF8&iwloc=&output=embed" frameBorder={0} allowFullScreen />
                </div>
                </div>
            </div>
        </body>
             
                   
    )
}
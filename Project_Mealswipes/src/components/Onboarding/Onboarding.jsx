import Header from '../Header/Header.jsx';
import './Onboarding.css';
import { Link } from "react-router-dom";


export default function Onboarding() {
    return (
        <body>
            <Header num={4} />
            <section id="section2">
                <div className="row section-header my-4 mx-auto">
                    <div className="col align-self-center">SIGN UP</div>
                </div>
                <div className="row volunteer">
                    <div className="col align-self-center">
                        <Link className="onelink" to='/volsignup'>VOLUNTEER</Link>
                    </div>
                </div>
                <div className="row donor">
                    <div className="col align-self-center">
                        <Link className="onelink" to='/donsignup'>DONOR</Link>
                    </div>
                </div>
                <div className="row shelter">
                    <div className="col align-self-center">
                        <Link className="onelink" to='/shelsignup'>SHELTER</Link>
                    </div>
                </div>
            </section>

        </body>
    )
}
import DHeader from '../Donor_Header/Donor_Header.jsx';
import './Donor_History.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faClock, faArrowCircleLeft } from '@fortawesome/free-solid-svg-icons';

import { Link } from "react-router-dom";

export default function Donor_History() {
    return (
        <body>
            <DHeader></DHeader>
            <section className="push"></section>
            <section className="trip-history">
                <div className="row">
                    <div className="col card mt-3">
                        
                        <div className="card-body">
                            <h5 className="card-title">DONOR HISTORY</h5>
                            <div className="row">
                                <table>
                                    <tbody><tr>
                                        <th>DATE</th>
                                        <th>DESCRIPTION</th>
                                        <th>RATING</th>
                                        <th>VOLUNTEER NAME</th>
                                        <th>PICK UP TIME</th>
                                    </tr>
                                        <tr>
                                            <td>07/02/2023</td>
                                            <td>BBQ Pizza</td>
                                            <td>4.5</td>
                                            <td>George</td>
                                            <td>12pm</td>
                                        </tr>
                                        <tr>
                                            <td>17/02/2023</td>
                                            <td>Spaguetti</td>
                                            <td>5</td>
                                            <td>Joan</td>
                                            <td>13.15pm</td>
                                        </tr>
                                        <tr>
                                            <td>21/02/2023</td>
                                            <td>Indian rice</td>
                                            <td>3.8</td>
                                            <td>Jack</td>
                                            <td>5pm</td>
                                        </tr>
                                    </tbody></table>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

        </body>
    )
}

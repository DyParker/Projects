import Header from '../Header/Header.jsx';
import chart from './Food_Waste_US_Chart.png';
import './About.css';



export default function about() {
  return (
    <body>
      <Header num={2} />
      <section id="aboutus">
        <div className="col section-header my-2 mx-0">FOOD WASTE IN THE U.S</div>
        <div className="row">
          <div className="col food-waste">
            <p>Food waste is an avoidable problem that results from inadequate resources put towards getting leftover food to people who need it most. We believe many people would be willing to dedicate their time and energy to helping their communities if we lower the information barrier, create social incentives, and raise awareness of food insecurity.</p>
          </div>
          <div className="col food-waste-pic">
            <img src={chart} alt={"piechart"} id="aboutuspic" />
          </div>
        </div>
        <div className="row mission-statement my-2">
          <p>Our mission is to reduce hunger and increase food sustainability in our communities.</p>
        </div>
      </section>
    </body>
  )
}
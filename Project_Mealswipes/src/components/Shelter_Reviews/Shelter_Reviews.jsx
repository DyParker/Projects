import SHeader from '../Shelter_Header/Shelter_Header.jsx';
import './Shelter_Reviews.css';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';


export default function Shelter_Reviews() {

  return (

    <body>
      <SHeader></SHeader>
      <section className="push"></section>
      <div>
        <div className="row backarrow"><a href="shelter.html"><i className="bi bi-arrow-left" /></a></div>
        <div className="row">
          <div className="col card mt-3">
            <div className="card-body">
              <h5 className="card-title">Leave a review about the volunteers</h5>
              <a className="btn btn-success btn-green" href="#reviews-anchor" id="open-review-box">Write Review</a>
              <div className="row" id="post-review-box" style={{ display: 'none' }}>
                <div className="col-md-12">
                  <form acceptCharset="UTF-8" action method="post">
                    <input id="ratings-hidden" name="rating" type="hidden" />
                    <textarea className="form-control animated" cols={50} id="new-review" name="comment" placeholder="Enter your review here..." rows={5} defaultValue={""} />
                    <div className="text-right">
                      <div className="stars starrr" data-rating={0} />
                      <a className="btn btn-danger btn-sm" href="#" id="close-review-box" style={{ display: 'none', marginRight: '10px' }}>
                        <span className="glyphicon glyphicon-remove" />Cancel</a>
                      <button className="btn btn-success btn-lg" type="submit">Save</button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </div>
          <div className="row">
            <div className="col card mt-3">
              <div className="card-body">
                <h5 className="card-title">Your evaluation</h5>
                <div className="col-xs-12 col-md-6">
                  <div className="well well-sm">
                    <div className="row">
                      <div className="col-xs-12 col-md-6 text-center">
                        <h1 className="rating-num">
                          4.0</h1>
                        <div className="container">
                          <span id="rateMe4" className="feedback" />
                        </div>
                        {/* rating.js file */}
                        <div>
                          <span className="bi bi-star-fill" />
                          <span className="bi bi-star-fill" />
                          <span className="bi bi-star-fill" />
                          <span className="bi bi-star-fill" />
                          <span className="bi bi-star" />
                        </div>
                        <div>1,050,008 total</div>
                      </div>
                      <div className="col-xs-12 col-md-6">
                        <div className="row rating-desc">
                          <div className="col-xs-3 col-md-3 text-right">
                            <span className="bi bi-star" />5
                          </div>
                          <div className="col-xs-8 col-md-9">
                            <div className="progress progress-striped">
                              <div className="progress-bar progress-bar-success" role="progressbar" aria-valuenow={20} aria-valuemin={0} aria-valuemax={100} style={{ width: '80%' }}>
                                <span className="sr-only">80%</span>
                              </div>
                            </div>
                          </div>
                          {/* end 5 */}
                          <div className="col-xs-3 col-md-3 text-right">
                            <span className="bi bi-star" />4
                          </div>
                          <div className="col-xs-8 col-md-9">
                            <div className="progress">
                              <div className="progress-bar progress-bar-success" role="progressbar" aria-valuenow={20} aria-valuemin={0} aria-valuemax={100} style={{ width: '60%' }}>
                                <span className="sr-only">60%</span>
                              </div>
                            </div>
                          </div>
                          {/* end 4 */}
                          <div className="col-xs-3 col-md-3 text-right">
                            <span className="bi bi-star" />3
                          </div>
                          <div className="col-xs-8 col-md-9">
                            <div className="progress">
                              <div className="progress-bar progress-bar-info" role="progressbar" aria-valuenow={20} aria-valuemin={0} aria-valuemax={100} style={{ width: '40%' }}>
                                <span className="sr-only">40%</span>
                              </div>
                            </div>
                          </div>
                          {/* end 3 */}
                          <div className="col-xs-3 col-md-3 text-right">
                            <span className="bi bi-star" />2
                          </div>
                          <div className="col-xs-8 col-md-9">
                            <div className="progress">
                              <div className="progress-bar progress-bar-warning" role="progressbar" aria-valuenow={20} aria-valuemin={0} aria-valuemax={100} style={{ width: '20%' }}>
                                <span className="sr-only">20%</span>
                              </div>
                            </div>
                          </div>
                          {/* end 2 */}
                          <div className="col-xs-3 col-md-3 text-right">
                            <span className="bi bi-star" />1
                          </div>
                          <div className="col-xs-8 col-md-9">
                            <div className="progress">
                              <div className="progress-bar progress-bar-danger" role="progressbar" aria-valuenow={80} aria-valuemin={0} aria-valuemax={100} style={{ width: '15%' }}>
                                <span className="sr-only">15%</span>
                              </div>
                            </div>
                          </div>
                          {/* end 1 */}
                        </div>
                        {/* end row */}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div className="row">
              <div className="col card mt-3">
                <div className="card-body">
                  <h5 className="card-title">Text Reviews</h5>
                </div>
                <div className="carousel-reviews broun-block">
                  <div className="container">
                    <div className="row">
                      <div id="carousel-reviews" className="carousel slide" data-ride="carousel">
                        <div className="carousel-inner">
                          <div className="item active">
                            <div className="col-md-4 col-sm-6">
                              <div className="block-text rel zmin">
                                <a title href="#">Joe</a>
                                <div className="mark">Rating:
                                  <span className="rating-input">
                                    <span data-value={0} className="bi bi-star-fill" />
                                    <span data-value={1} className="bi bi-star-fill" />
                                    <span data-value={2} className="bi bi-star-fill" />
                                    <span data-value={3} className="bi bi-star-fill" />
                                    <span data-value={4} className="bi bi-star-fill" />
                                    <span data-value={5} className="bi bi-star-fill" />
                                  </span>
                                </div>
                                <p>Never before has there been a good film portrayal of ancient Greece's favourite myth. So why would Hollywood start now? This latest attempt at bringing the son of Zeus to the big screen is brought to us by X-Men: The last Stand director Brett Ratner. If the name of the director wasn't enough to dissuade ...</p>
                                <ins className="ab zmin sprite sprite-i-triangle block" />
                              </div>
                            </div>
                            <div className="col-md-4 col-sm-6 hidden-xs">
                              <div className="block-text rel zmin">
                                <a title href="#">Chelsea</a>
                                <div className="mark">Rating:
                                  <span className="rating-input">
                                    <span data-value={0} className="bi bi-star-fill" />
                                    <span data-value={1} className="bi bi-star-fill" />
                                    <span data-value={2} className="bi bi-star-fill" />
                                    <span data-value={3} className="bi bi-star-half" />
                                    <span data-value={4} className="bi bi-star" />
                                    <span data-value={5} className="bi bi-star" />
                                  </span>
                                </div>
                                <p>The 2013 movie "The Purge" left a bad taste in all of our mouths as nothing more than a pseudo-slasher with a hamfisted plot, poor pacing, and a desperate attempt at "horror." Upon seeing the first trailer for "The Purge: Anarchy," my first and most immediate thought was "we really don't need another one of these."</p>
                                <ins className="ab zmin sprite sprite-i-triangle block" />
                              </div>
                            </div>
                            <div className="col-md-4 col-sm-6 hidden-sm hidden-xs">
                              <div className="block-text rel zmin">
                                <a title href="#">Ana</a>
                                <div className="mark">Rating:
                                  <span className="rating-input">
                                    <span data-value={0} className="bi bi-star-fill" />
                                    <span data-value={1} className="bi bi-star-fill" />
                                    <span data-value={2} className="bi bi-star-fill" />
                                    <span data-value={3} className="bi bi-star-fill" />
                                    <span data-value={4} className="bi bi-star" />
                                    <span data-value={5} className="bi bi-star" />
                                  </span>
                                </div>
                                <p>What a funny and entertaining film! I did not know what to expect, this is the fourth film in this vehicle's universe with the two Cars movies and then the first Planes movie. I was wondering if maybe Disney pushed it a little bit. However, Planes: Fire and Rescue is an entertaining film that is a fantastic sequel in this magical franchise. </p>
                                <ins className="ab zmin sprite sprite-i-triangle block" />
                              </div>
                            </div>
                          </div>
                        </div>
                        <a className="left carousel-control" href="#carousel-reviews" role="button" data-slide="prev">
                          <span className="glyphicon glyphicon-chevron-left" />
                        </a>
                        <a className="right carousel-control" href="#carousel-reviews" role="button" data-slide="next">
                          <span className="glyphicon glyphicon-chevron-right" />
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div></div></div>


    </body>

  )
}
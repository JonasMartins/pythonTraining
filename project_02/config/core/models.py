from django.db import models

# Create your models here.
# models.Model pattern Django model class for all models
class Movie(models.Model):
  NOT_RATED = 0
  RATED_G = 1
  RATED_PG = 2
  RATED_R = 3
  RATINGS = (
      (NOT_RATED, 'NR - Not Rated'),
      (RATED_G,
        'G - General Audiences'),
      (RATED_PG,
        'PG - Parental Guidance '
        'Suggested'),
      (RATED_R, 'R - Restricted'),
  )
  # title like a varchar field with max characters
  title = models.CharField(
      max_length=140)
  # Like a description, a breif film's history, with no 
  # max limit charcters size 
  plot = models.TextField()
  # movie's year, an integer field
  year = models.PositiveIntegerField()
  # integer field, more complex choices is an optional
  # argument that any fild has, receving an integer value
  rating = models.IntegerField(
      choices=RATINGS,
      default=NOT_RATED)
  runtime = \
      models.PositiveIntegerField()
  website = models.URLField(
      blank=True)
  # This method helps django to cenvert the object model
  # to a string
  def __str__(self):
    return '{} ({})'.format(
      self.title, self.year)
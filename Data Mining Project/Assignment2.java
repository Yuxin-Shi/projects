import java.io.IOException;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;

// If you are looking for Java data structures, these are highly useful.
// Remember that an important part of your mark is for doing as much in SQL (not Java) as you can.
// Solutions that use only or mostly Java will not receive a high mark.
import java.util.ArrayList;
//import java.util.Map;
import java.util.HashMap;
//import java.util.Set;
//import java.util.HashSet;
public class Assignment2 extends JDBCSubmission {

    public Assignment2() throws ClassNotFoundException {

        Class.forName("org.postgresql.Driver");
    }

    Connection connection;

    @Override
    public boolean connectDB(String url, String username, String password){
        try{
            this.connection = DriverManager.getConnection(url, username, password);
            Statement statement = connection.createStatement();
            try {
                statement.execute("SET search_path TO parlgov");
            }
            finally { statement.close(); }

            return connection != null && !connection.isClosed();
        }
        catch (SQLException se)
        {
            System.err.println("SQL Exception." +
                    "<Message>: " + se.getMessage());
        }
        return false;
    }

    @Override
    public boolean disconnectDB() {
        connection = null;
        return connection == null;
    }

    @Override
    public ElectionCabinetResult electionSequence(String countryName) {
        List<Integer> e_list = new ArrayList<Integer>();
        List<Integer> c_list = new ArrayList<Integer>();
        ElectionCabinetResult er = new ElectionCabinetResult(e_list, c_list);
        try{
            String queryString =
                "SELECT cabinet_id, election_id\n"
              + "FROM(\n"
              + "SELECT cabinet_id, a1.election_id, country.name as country_name\n"
              + "FROM ((SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, election_id, e_date, e_type\n"
              + "           FROM election JOIN cabinet ON election_id = election.id\n"
              + "           ORDER BY election_id) UNION ALL (SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, t1.id as election_id, after_date as e_date, t1.e_type\n"
              + "                                                                  FROM  cabinet JOIN \n"
              + "                                                                              (SELECT e2.id, e2.country_id, e1.e_date as before_date, e2.e_date as after_date, e2.e_type\n"
              + "                                                                               FROM election e1 JOIN election e2 ON e1.previous_ep_election_id = e2.id\n"
              + "                                                                               WHERE e1.e_type = 'European Parliament' and e2.e_type = 'European Parliament'\n"
              + "                                                                               ) as t1 ON cabinet.country_id = t1.country_id\n"
              + "                                                                  WHERE cabinet.start_date >= after_date and cabinet.start_date <= before_date\n"
              + "                                                                  ) UNION ALL\n"
              + "                                                                 (SELECT cabinet.id as cabinet_id, cabinet.start_date as cabinet_start_date, t2.id, e_date, t2.e_type\n"
              + "                                                                  FROM cabinet JOIN \n"
              + "                                                                             (SELECT id, country_id, e_date, e_type FROM election\n"
              + "                                                                              WHERE id NOT IN (SELECT previous_ep_election_id FROM election \n"
              + "     \t\t                                                                            WHERE e_type = 'European Parliament' and previous_ep_election_id is NOT NULL) \n"
              + "                                                                              and e_type = 'European Parliament'\n"
              + "                                                                              ) as t2 ON cabinet.country_id = t2.country_id\n"
              + "                                                                  WHERE cabinet.start_date >= t2.e_date)) as a1 LEFT JOIN cabinet ON a1.cabinet_id = cabinet.id \n"
              + "\t\t\t\t\t\t\t\t                                                                                                LEFT JOIN  country ON country.id = cabinet.country_id\n"
              + "ORDER BY e_date DESC, cabinet_id, CASE e_type WHEN 'Parliamentary election' THEN 1 WHEN 'European Parliament' THEN 2 END\n"
              + ") as a2\n"
              + "WHERE country_name = " + countryName;
            PreparedStatement ps = connection.prepareStatement(queryString);
            ResultSet rs = ps.executeQuery();
            while (rs.next()){
                er.elections.add(rs.getInt(2));
                er.cabinets.add(rs.getInt(1));
            }
        }
        catch (SQLException se)
        {
            System.err.println("SQL Exception." +
                    "<Message>: " + se.getMessage());
        }
        return er;
    }

    @Override
    public List<Integer> findSimilarPoliticians(Integer politicianName, Float threshold) {
        List<Integer> result = new ArrayList<>();
        try{
            String queryString = "select id, description from politician_president";
            PreparedStatement ps = connection.prepareStatement(queryString);
            ResultSet rs = ps.executeQuery();
            HashMap<Integer,String> politicians = new HashMap<Integer,String>();
            while (rs.next()) {
                politicians.put(rs.getInt(1), rs.getString(2));
            }
            String description = politicians.get(politicianName);

            for (int key : politicians.keySet()){
                if (super.similarity(description, politicians.get(key)) >= threshold &&
                        key != politicianName){
                    result.add(key);
                }
            }
        }
        catch (SQLException se)
        {
            System.err.println("SQL Exception." +
                    "<Message>: " + se.getMessage());
        }

        return result;
    }

    public static void main(String[] args) throws Exception{
        Assignment2 a2 = new Assignment2();
        a2.connectDB("jdbc:postgresql://localhost:5432/postgres", "postgres", "");
        float f = (float) 0.5;
        System.out.println(a2.findSimilarPoliticians(9,f));
        System.out.println(a2.electionSequence("'France'"));
    }

}

